#define PY_SSIZE_T_CLEAN
#define NOMINMAX
#include <Python.h>

#include <Windows.h>

#include <carb/Framework.h>
#include <carb/Interface.h>

#include <PxParticleBuffer.h>
#include <PxPBDParticleSystem.h>
#include <cudamanager/PxCudaContext.h>
#include <cudamanager/PxCudaContextManager.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace
{

// Isaac Sim 6.0 ships OpenUSD 25.11.  SdfPath is deliberately an 8-byte
// value type in that ABI.  We construct it through the exact exported
// constructor instead of linking against an OpenUSD SDK that Isaac does not
// ship.  Paths are cached for the module lifetime, so their node references
// remain valid without guessing the private inline destructor implementation.
struct OpaqueSdfPath
{
    std::uint64_t storage = 0;
};
static_assert(sizeof(OpaqueSdfPath) == 8, "Unexpected SdfPath ABI storage size");

using SdfPathStringConstructor = void(__fastcall*)(OpaqueSdfPath*, const std::string&);

constexpr const char* kSdfPathConstructorSymbol =
    "??0SdfPath@pxrInternal_v0_25_11__pxrReserved__@@QEAA@AEBV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@@Z";

// Prefix of omni::physx::IPhysx 5.0.  These are the first three entries in
// NVIDIA's public IPhysx interface and therefore do not depend on later table
// additions.
struct IPhysxPrefix
{
    std::size_t(CARB_ABI* getObjectId)(const OpaqueSdfPath&, int type);
    void*(CARB_ABI* getPhysXPtr)(const OpaqueSdfPath&, int type);
    void*(CARB_ABI* getPhysXPtrFast)(std::size_t objectId);
};

// Prefix of omni::physx::IPhysxPrivate 2.0.  getCudaContextManager is the
// third entry in the installed/public interface.
struct IPhysxPrivatePrefix
{
    void* getPhysXScene;
    void* getRigidBodyInstancedData;
    physx::PxCudaContextManager*(CARB_ABI* getCudaContextManager)();
};

constexpr int kParticleSetType = 12; // omni::physx::ePTParticleSet
constexpr int kParticleSystemType = 11; // omni::physx::ePTParticleSystem
constexpr std::uint32_t kMaximumSanePrimaryParticles = 100'000'000;
constexpr std::uint32_t kMaximumSaneDiffuseParticles = 100'000'000;
constexpr const char* kBridgeSchema = "physx_diffuse_native_bridge/2";
constexpr const char* kPhysXSourceTag = "110.1-omni-and-physx-5.9.0";
constexpr const char* kPhysXSourceCommit = "517a0073715120e114ee055b63b26c95e00d9039";

carb::Framework* acquireFramework()
{
    HMODULE carbModule = GetModuleHandleW(L"carb.dll");
    if (!carbModule)
        throw std::runtime_error("carb.dll is not loaded; import inside a running Kit/Isaac process");

    using AcquireFrameworkFn = carb::Framework*(CARB_ABI*)(const char*, carb::Version);
    auto fn = reinterpret_cast<AcquireFrameworkFn>(GetProcAddress(carbModule, "acquireFramework"));
    if (!fn)
        throw std::runtime_error("carb.dll does not export acquireFramework");

    carb::Framework* framework = fn("swamp.physx_diffuse_bridge", carb::kFrameworkVersion);
    if (!framework)
        throw std::runtime_error("Carbonite framework acquisition failed");
    return framework;
}

template <typename T>
T* acquireInterface(const char* name, carb::Version version)
{
    carb::Framework* framework = acquireFramework();
    auto* result = framework->tryAcquireInterfaceWithClient(
        "swamp.physx_diffuse_bridge", carb::InterfaceDesc{ name, version }, "omni.physx.plugin");
    if (!result)
        throw std::runtime_error(std::string("Unable to acquire Carbonite interface ") + name);
    return static_cast<T*>(result);
}

SdfPathStringConstructor getSdfPathConstructor()
{
    HMODULE sdfModule = GetModuleHandleW(L"usd_sdf.dll");
    if (!sdfModule)
        throw std::runtime_error("usd_sdf.dll is not loaded; enable omni.physx before importing the bridge");
    auto constructor = reinterpret_cast<SdfPathStringConstructor>(
        GetProcAddress(sdfModule, kSdfPathConstructorSymbol));
    if (!constructor)
        throw std::runtime_error(
            "OpenUSD SdfPath ABI mismatch: the OpenUSD 25.11 constructor symbol was not found");
    return constructor;
}

const OpaqueSdfPath& cachedSdfPath(const std::string& text)
{
    static std::mutex mutex;
    static std::unordered_map<std::string, std::unique_ptr<OpaqueSdfPath>> paths;
    std::lock_guard<std::mutex> lock(mutex);
    auto found = paths.find(text);
    if (found != paths.end())
        return *found->second;

    auto path = std::make_unique<OpaqueSdfPath>();
    getSdfPathConstructor()(path.get(), text);
    auto [inserted, ok] = paths.emplace(text, std::move(path));
    if (!ok)
        throw std::runtime_error("Unable to cache SdfPath");
    return *inserted->second;
}

physx::PxParticleAndDiffuseBuffer* resolveBuffer(const char* pathText)
{
    if (!pathText || pathText[0] != '/')
        throw std::runtime_error("particle_set_path must be an absolute USD path");
    static IPhysxPrefix* physxInterface =
        acquireInterface<IPhysxPrefix>("omni::physx::IPhysx", carb::Version{ 5, 0 });
    void* pointer = physxInterface->getPhysXPtr(cachedSdfPath(pathText), kParticleSetType);
    if (!pointer)
        throw std::runtime_error(
            std::string("No live PhysX particle-set buffer exists at USD path ") + pathText);
    return static_cast<physx::PxParticleAndDiffuseBuffer*>(pointer);
}

physx::PxPBDParticleSystem* resolveParticleSystem(const char* pathText)
{
    if (!pathText || pathText[0] != '/')
        throw std::runtime_error("particle_system_path must be an absolute USD path");
    static IPhysxPrefix* physxInterface =
        acquireInterface<IPhysxPrefix>("omni::physx::IPhysx", carb::Version{ 5, 0 });
    void* pointer = physxInterface->getPhysXPtr(cachedSdfPath(pathText), kParticleSystemType);
    if (!pointer)
        throw std::runtime_error(
            std::string("No live PhysX particle system exists at USD path ") + pathText);
    return static_cast<physx::PxPBDParticleSystem*>(pointer);
}

void setPythonError(const std::exception& error)
{
    PyErr_SetString(PyExc_RuntimeError, error.what());
}

struct PythonDeleter
{
    void operator()(PyObject* value) const
    {
        Py_XDECREF(value);
    }
};
using UniquePyObject = std::unique_ptr<PyObject, PythonDeleter>;

PyObject* abiInfo(PyObject*, PyObject*)
{
    return Py_BuildValue(
        "{s:s,s:i,s:i,s:i,s:i,s:s,s:s,s:s}",
        "bridge_schema", kBridgeSchema,
        "iphysx_major", 5,
        "iphysx_minor", 0,
        "iphysx_private_major", 2,
        "iphysx_private_minor", 0,
        "openusd_abi", "25.11 (pxrInternal_v0_25_11__pxrReserved__)",
        "physx_source_tag", kPhysXSourceTag,
        "physx_source_commit", kPhysXSourceCommit);
}

PyObject* probe(PyObject*, PyObject* args)
{
    const char* pathText = nullptr;
    if (!PyArg_ParseTuple(args, "s:probe", &pathText))
        return nullptr;
    try
    {
        auto* buffer = resolveBuffer(pathText);
        const std::uint32_t activeCount = buffer->getNbActiveDiffuseParticles();
        const std::uint32_t maximumCount = buffer->getMaxDiffuseParticles();
        if (activeCount > maximumCount || maximumCount > kMaximumSaneDiffuseParticles)
            throw std::runtime_error(
                "Diffuse buffer count sanity check failed; refusing an unsafe ABI result "
                "(active_count=" + std::to_string(activeCount) +
                ", max_count=" + std::to_string(maximumCount) +
                ", sanity_limit=" + std::to_string(kMaximumSaneDiffuseParticles) + ")");
        return Py_BuildValue(
            "{sI,sI,sK,sK}",
            "active_count", activeCount,
            "max_count", maximumCount,
            "position_lifetime_device_ptr",
            static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(buffer->getDiffusePositionLifeTime())),
            "velocity_device_ptr",
            static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(buffer->getDiffuseVelocities())));
    }
    catch (const std::exception& error)
    {
        setPythonError(error);
        return nullptr;
    }
}

PyObject* particleSystemInfo(PyObject*, PyObject* args)
{
    const char* pathText = nullptr;
    if (!PyArg_ParseTuple(args, "s:particle_system_info", &pathText))
        return nullptr;
    try
    {
        auto* system = resolveParticleSystem(pathText);
        const std::uint32_t flags = static_cast<std::uint32_t>(system->getParticleFlags());
        const float particleContactOffset = system->getParticleContactOffset();
        return Py_BuildValue(
            "{sI,sf,sf,sf,sf,sO}",
            "particle_flags", flags,
            "particle_contact_offset", particleContactOffset,
            // PxgParticleSystemCore::updateParticleSystemData() assigns the
            // Diffuse kernel radius as 2 * particleContactOffset.
            "diffuse_neighbor_radius", 2.0f * particleContactOffset,
            "rest_offset", system->getRestOffset(),
            "fluid_rest_offset", system->getFluidRestOffset(),
            "full_diffuse_advection",
            (flags & static_cast<std::uint32_t>(physx::PxParticleFlag::eFULL_DIFFUSE_ADVECTION))
                ? Py_True
                : Py_False);
    }
    catch (const std::exception& error)
    {
        setPythonError(error);
        return nullptr;
    }
}

PyObject* setFullDiffuseAdvection(PyObject*, PyObject* args)
{
    const char* pathText = nullptr;
    int enabled = 0;
    if (!PyArg_ParseTuple(args, "sp:set_full_diffuse_advection", &pathText, &enabled))
        return nullptr;
    try
    {
        auto* system = resolveParticleSystem(pathText);
        system->setParticleFlag(physx::PxParticleFlag::eFULL_DIFFUSE_ADVECTION, enabled != 0);
        Py_RETURN_NONE;
    }
    catch (const std::exception& error)
    {
        setPythonError(error);
        return nullptr;
    }
}

class ContextGuard
{
public:
    explicit ContextGuard(physx::PxCudaContextManager* manager) : mManager(manager)
    {
        mManager->acquireContext();
    }
    ~ContextGuard()
    {
        mManager->releaseContext();
    }

private:
    physx::PxCudaContextManager* mManager;
};

PyObject* readDiffuse(PyObject*, PyObject* args)
{
    const char* pathText = nullptr;
    if (!PyArg_ParseTuple(args, "s:read_diffuse", &pathText))
        return nullptr;
    try
    {
        auto* buffer = resolveBuffer(pathText);
        const std::uint32_t activeCount = buffer->getNbActiveDiffuseParticles();
        const std::uint32_t maximumCount = buffer->getMaxDiffuseParticles();
        if (activeCount > maximumCount || maximumCount > kMaximumSaneDiffuseParticles)
            throw std::runtime_error("Diffuse buffer count sanity check failed; refusing an unsafe copy");

        constexpr std::size_t stride = sizeof(physx::PxVec4);
        if (activeCount > std::numeric_limits<Py_ssize_t>::max() / stride)
            throw std::runtime_error("Diffuse byte size exceeds Python's addressable buffer size");
        const Py_ssize_t byteCount = static_cast<Py_ssize_t>(activeCount * stride);

        UniquePyObject positions(PyBytes_FromStringAndSize(nullptr, byteCount));
        UniquePyObject velocities(PyBytes_FromStringAndSize(nullptr, byteCount));
        if (!positions || !velocities)
            return nullptr;

        if (activeCount > 0)
        {
            static IPhysxPrivatePrefix* privateInterface = acquireInterface<IPhysxPrivatePrefix>(
                "omni::physx::IPhysxPrivate", carb::Version{ 2, 0 });
            physx::PxCudaContextManager* manager = privateInterface->getCudaContextManager();
            if (!manager || !manager->getCudaContext())
                throw std::runtime_error("PhysX CUDA context manager is unavailable");

            ContextGuard context(manager);
            physx::PxCudaContext* cuda = manager->getCudaContext();
            const physx::PxCUresult positionResult = cuda->memcpyDtoH(
                PyBytes_AS_STRING(positions.get()),
                reinterpret_cast<CUdeviceptr>(buffer->getDiffusePositionLifeTime()),
                static_cast<std::size_t>(byteCount));
            const physx::PxCUresult velocityResult = cuda->memcpyDtoH(
                PyBytes_AS_STRING(velocities.get()),
                reinterpret_cast<CUdeviceptr>(buffer->getDiffuseVelocities()),
                static_cast<std::size_t>(byteCount));
            if (positionResult != 0 || velocityResult != 0)
                throw std::runtime_error(
                    "CUDA device-to-host copy failed (position=" + std::to_string(positionResult) +
                    ", velocity=" + std::to_string(velocityResult) + ")");
        }

        PyObject* result = Py_BuildValue(
            "{sI,sI,si,sO,sO}",
            "active_count", activeCount,
            "max_count", maximumCount,
            "stride_floats", 4,
            "position_lifetime", positions.get(),
            "velocity", velocities.get());
        return result;
    }
    catch (const std::exception& error)
    {
        setPythonError(error);
        return nullptr;
    }
}

PyObject* readParticleFrame(PyObject*, PyObject* args)
{
    const char* pathText = nullptr;
    if (!PyArg_ParseTuple(args, "s:read_particle_frame", &pathText))
        return nullptr;
    try
    {
        auto* buffer = resolveBuffer(pathText);
        const std::uint32_t primaryCount = buffer->getNbActiveParticles();
        const std::uint32_t primaryMaximum = buffer->getMaxParticles();
        const std::uint32_t diffuseCount = buffer->getNbActiveDiffuseParticles();
        const std::uint32_t diffuseMaximum = buffer->getMaxDiffuseParticles();
        if (primaryCount > primaryMaximum || primaryMaximum > kMaximumSanePrimaryParticles)
            throw std::runtime_error("Primary particle count sanity check failed; refusing an unsafe copy");
        if (diffuseCount > diffuseMaximum || diffuseMaximum > kMaximumSaneDiffuseParticles)
            throw std::runtime_error("Diffuse particle count sanity check failed; refusing an unsafe copy");

        constexpr std::size_t stride = sizeof(physx::PxVec4);
        if (primaryCount > std::numeric_limits<Py_ssize_t>::max() / stride ||
            diffuseCount > std::numeric_limits<Py_ssize_t>::max() / stride)
            throw std::runtime_error("Particle byte size exceeds Python's addressable buffer size");
        const Py_ssize_t primaryBytes = static_cast<Py_ssize_t>(primaryCount * stride);
        const Py_ssize_t diffuseBytes = static_cast<Py_ssize_t>(diffuseCount * stride);

        UniquePyObject primaryPositions(PyBytes_FromStringAndSize(nullptr, primaryBytes));
        UniquePyObject primaryVelocities(PyBytes_FromStringAndSize(nullptr, primaryBytes));
        UniquePyObject diffusePositions(PyBytes_FromStringAndSize(nullptr, diffuseBytes));
        UniquePyObject diffuseVelocities(PyBytes_FromStringAndSize(nullptr, diffuseBytes));
        if (!primaryPositions || !primaryVelocities || !diffusePositions || !diffuseVelocities)
            return nullptr;

        static IPhysxPrivatePrefix* privateInterface = acquireInterface<IPhysxPrivatePrefix>(
            "omni::physx::IPhysxPrivate", carb::Version{ 2, 0 });
        physx::PxCudaContextManager* manager = privateInterface->getCudaContextManager();
        if (!manager || !manager->getCudaContext())
            throw std::runtime_error("PhysX CUDA context manager is unavailable");

        ContextGuard context(manager);
        physx::PxCudaContext* cuda = manager->getCudaContext();
        auto copy = [cuda](PyObject* destination, const void* source, std::size_t bytes, const char* label)
        {
            if (bytes == 0)
                return;
            const physx::PxCUresult result = cuda->memcpyDtoH(
                PyBytes_AS_STRING(destination), reinterpret_cast<CUdeviceptr>(source), bytes);
            if (result != 0)
                throw std::runtime_error(
                    std::string("CUDA device-to-host copy failed for ") + label +
                    " (result=" + std::to_string(result) + ")");
        };
        copy(primaryPositions.get(), buffer->getPositionInvMasses(), primaryBytes, "primary positions");
        copy(primaryVelocities.get(), buffer->getVelocities(), primaryBytes, "primary velocities");
        copy(diffusePositions.get(), buffer->getDiffusePositionLifeTime(), diffuseBytes, "diffuse positions");
        copy(diffuseVelocities.get(), buffer->getDiffuseVelocities(), diffuseBytes, "diffuse velocities");

        return Py_BuildValue(
            "{sI,sI,sI,sI,si,sO,sO,sO,sO}",
            "primary_active_count", primaryCount,
            "primary_max_count", primaryMaximum,
            "diffuse_active_count", diffuseCount,
            "diffuse_max_count", diffuseMaximum,
            "stride_floats", 4,
            "primary_position_inv_mass", primaryPositions.get(),
            "primary_velocity", primaryVelocities.get(),
            "diffuse_position_lifetime", diffusePositions.get(),
            "diffuse_velocity", diffuseVelocities.get());
    }
    catch (const std::exception& error)
    {
        setPythonError(error);
        return nullptr;
    }
}

PyMethodDef methods[] = {
    { "abi_info", abiInfo, METH_NOARGS, "Report the bridge's pinned Isaac/PhysX ABI contract." },
    { "probe", probe, METH_VARARGS, "Resolve a live particle set and report native Diffuse buffer metadata." },
    { "particle_system_info", particleSystemInfo, METH_VARARGS, "Report the native PBD particle-system flags and offsets." },
    { "set_full_diffuse_advection", setFullDiffuseAdvection, METH_VARARGS, "Set PhysX's native full-neighbour Diffuse advection flag between simulation steps." },
    { "read_diffuse", readDiffuse, METH_VARARGS, "Copy active native Diffuse position/lifetime and velocity arrays to Python bytes." },
    { "read_particle_frame", readParticleFrame, METH_VARARGS, "Copy primary and Diffuse native arrays from one fetched frame." },
    { nullptr, nullptr, 0, nullptr },
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_physx_diffuse_bridge",
    "Isaac Sim 6.0 native PhysX Diffuse readback bridge.",
    -1,
    methods,
};

} // namespace

PyMODINIT_FUNC PyInit__physx_diffuse_bridge()
{
    return PyModule_Create(&module);
}
