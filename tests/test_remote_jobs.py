import importlib.util
import json
from pathlib import Path
import unittest
from unittest import mock
import tempfile

spec=importlib.util.spec_from_file_location('remote_jobs',Path(__file__).resolve().parents[1]/'tools/remote_jobs.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class RemoteJobsTests(unittest.TestCase):
    def task(self):
        return dict(schema_version=1,kind='agent_request',id='20260912_001',order=1,title='测试',request='检查',parameters={},depends_on=[],on_failure='stop_queue')

    def test_valid(self):
        t=self.task();self.assertEqual(module.validate(json.dumps(t)),t)

    def test_reject_bad_requests(self):
        for change in [dict(id='../oops'),dict(id=123),dict(order=True),dict(order=-1),dict(order=None),dict(command='run something'),dict(depends_on=['a','a']),dict(depends_on=['20260912_001']),dict(parameters=[]),dict(on_failure='ignore'),dict(request='')]:
            t={**self.task(),**change}
            with self.assertRaises(ValueError):module.validate(json.dumps(t))

    def pipeline(self):
        return {**self.task(),'kind':'pipeline_job','scene':'static_water',
                'code':dict(repository='/data/repo',commit='a'*40,worktree_mode='detached'),
                'output':dict(directory='/data/jiachen/job_outputs/test'),
                'resources':dict(gpu_count=1),
                'execution':dict(mode='simulate_only',steps=[dict(name='sim',kind='simulation',uses_gpu=True,argv=['/data/wrapper.py','{worktree}/sim.py'])])}

    def test_pipeline(self):
        t=self.pipeline();self.assertEqual(module.validate(json.dumps(t)),t)

    def test_reject_placeholder_and_gpu_override(self):
        t=self.pipeline();t['code']['commit']='0'*40
        with self.assertRaises(ValueError):module.validate(json.dumps(t))
        t=self.pipeline();t['execution']['steps'][0]['env']={'SIM_GPU':'0'}
        with self.assertRaises(ValueError):module.validate(json.dumps(t))
        t=self.pipeline();t['execution']['mode']='simulate_and_render'
        with self.assertRaises(ValueError):module.validate(json.dumps(t))

    def test_dry_run_does_not_connect(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'task.json';path.write_text(json.dumps(self.pipeline()))
            with mock.patch('sys.argv',['remote_jobs','submit',str(path),'--dry-run']), mock.patch.object(module,'ssh') as ssh, mock.patch.object(module.subprocess,'run') as run:
                module.main();ssh.assert_not_called();run.assert_not_called()

    def test_submit_stages_part_and_server_validates(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'task.json';path.write_text(json.dumps(self.pipeline()))
            with mock.patch('sys.argv',['remote_jobs','submit',str(path)]), mock.patch.object(module,'ssh') as ssh, mock.patch.object(module.subprocess,'run') as run:
                module.main()
                self.assertTrue(run.call_args.args[0][-1].endswith('.json.part'))
                code=ssh.call_args.args[0]
                compile(code,'remote_publish','exec')
                self.assertIn('module.validate_job',code)
                self.assertIn('os.link(src,ready)',code)
                self.assertNotIn('os.link(src,archive)',code)


if __name__=='__main__':unittest.main()
