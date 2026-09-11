"""Submit immutable jobs to the existing sglab queue over SSH/SCP."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import uuid

HOST='sglab'
ROOT='/data/jiachen/jobs'


def validate(raw):
    data=json.loads(raw)
    if not isinstance(data,dict):raise ValueError('Task must be a JSON object')
    if data.get('schema_version')!=1:raise ValueError('schema_version must be 1')
    if not isinstance(data.get('id'),str) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}',data['id']):
        raise ValueError('Invalid id; use letters, numbers, underscores or hyphens')
    for field in ('title','request'):
        if not isinstance(data.get(field),str) or not data[field].strip():raise ValueError('Missing '+field)
    if data.get('kind') not in ('agent_request','pipeline_job'):raise ValueError('Unsupported kind')
    order=data.get('order')
    if type(order) is not int or order<0:raise ValueError('order must be a non-negative integer')
    if data.get('on_failure') not in ('stop_queue','continue'):raise ValueError('Explicit on_failure required')
    if not isinstance(data.get('parameters'),dict):raise ValueError('parameters must be an object')
    deps=data.get('depends_on')
    if not isinstance(deps,list) or any(not isinstance(x,str) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}',x) for x in deps):
        raise ValueError('depends_on must contain valid task IDs')
    if data['id'] in deps:raise ValueError('Task cannot depend on itself')
    if len(set(deps))!=len(deps):raise ValueError('Duplicate dependencies')
    if any(k in data for k in ('command','shell','script')):raise ValueError('Use execution.steps[].argv, not top-level commands')
    if data['kind']=='pipeline_job':
        code=data.get('code')
        if not isinstance(code,dict):raise ValueError('code must be an object')
        commit=code.get('commit')
        if not isinstance(commit,str) or not re.fullmatch(r'[0-9a-fA-F]{40}',commit) or int(commit,16)==0:
            raise ValueError('Use a real full 40-character commit SHA, not the template placeholder')
        for value in (code.get('repository'),data.get('output',{}).get('directory') if isinstance(data.get('output'),dict) else None):
            if not isinstance(value,str) or not value.startswith('/'):raise ValueError('Repository and output must be absolute server paths')
        if code.get('worktree_mode','detached')!='detached':raise ValueError('Client requires detached worktree mode')
        if not isinstance(data.get('scene'),str) or not data['scene'].strip():raise ValueError('scene required')
        if not isinstance(data.get('resources'),dict) or type(data['resources'].get('gpu_count')) is not int or data['resources']['gpu_count']!=1:
            raise ValueError('Exactly one GPU per job required')
        execution=data.get('execution')
        if not isinstance(execution,dict) or execution.get('mode') not in ('simulate_only','simulate_and_render'):
            raise ValueError('Invalid execution mode')
        steps=execution.get('steps')
        if not isinstance(steps,list) or not steps:raise ValueError('Explicit argv steps required')
        kinds=[]
        for step in steps:
            if not isinstance(step,dict):raise ValueError('Step must be an object')
            argv=step.get('argv')
            if not isinstance(argv,list) or not argv or any(not isinstance(x,str) or not x or '\0' in x for x in argv):
                raise ValueError('argv must be a non-empty string array')
            if Path(argv[0]).name in ('sh','bash','zsh','fish','sudo'):raise ValueError('Do not invoke a shell or sudo')
            if type(step.get('uses_gpu')) is not bool:raise ValueError('uses_gpu must be boolean')
            env=step.get('env',{})
            if not isinstance(env,dict) or any(not isinstance(k,str) or not isinstance(v,str) for k,v in env.items()):raise ValueError('Invalid step env')
            if {'SIM_GPU','CUDA_VISIBLE_DEVICES'} & env.keys():raise ValueError('Do not override queue GPU assignment')
            kinds.append(step.get('kind'))
        if 'simulation' not in kinds:raise ValueError('simulation step required')
        if ('render' in kinds)!=(execution['mode']=='simulate_and_render'):raise ValueError('render steps must match execution mode')
    return data


def ssh(code,*args):
    command='python3 -c '+shlex.quote(code)+' '+ ' '.join(shlex.quote(str(x)) for x in args)
    return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',HOST,command],check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    send=sub.add_parser('submit');send.add_argument('task',type=Path);send.add_argument('--dry-run',action='store_true')
    sub.add_parser('status')
    args=parser.parse_args()
    if args.action=='status':
        ssh("import subprocess; subprocess.run(['/data/jiachen/jobs/bin/job_queue.py','status'],check=True)");return
    raw=args.task.read_bytes()
    if len(raw)>1024*1024:raise ValueError('Task request must be <= 1 MiB; artifacts go elsewhere')
    data=validate(raw.decode('utf-8'));digest=hashlib.sha256(raw).hexdigest()
    if args.dry_run:print(json.dumps(dict(local_checks_passed=True,server_validated=False,id=data['id'],sha256=digest),indent=2));return
    # Only *.part is ignored by the live consumer. The server owns submitted/.
    token=uuid.uuid4().hex
    remote=f'{ROOT}/.incoming/{data["id"]}.{token}.json.part'
    subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=10',str(args.task.resolve()),HOST+':'+remote],check=True)
    ssh('''from pathlib import Path
import hashlib,importlib.util,json,os,sys
root=Path('/data/jiachen/jobs');src=Path(sys.argv[1]);job_id=sys.argv[2];expected=sys.argv[3]
if src.parent!=root/'.incoming' or src.suffix!='.part':raise ValueError('Invalid staging path')
raw=src.read_bytes()
if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Upload checksum mismatch')
spec=importlib.util.spec_from_file_location('job_queue',root/'bin/job_queue.py')
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
job=json.loads(raw)
errors=module.validate_job(job,job_id+'.json')
if errors:raise ValueError('Server validation failed: '+str(errors))
if any((root/f/(job_id+'.json')).exists() for f in ('submitted','pending','running','done','failed','blocked','invalid')):
 raise ValueError('ID already submitted; inspect status instead of retrying blindly')
ready=root/'.incoming'/(job_id+'.json')
os.link(src,ready) # Atomic no-overwrite publication; consumer owns ledger and pending.
src.unlink()
print('Uploaded and server-validated '+job_id+' sha256='+expected+'; awaiting queue ingestion')
''',remote,data['id'],digest)


if __name__=='__main__':main()
