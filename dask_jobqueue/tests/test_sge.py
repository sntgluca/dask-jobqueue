from time import sleep, time

import pytest
from distributed import Client

from dask_jobqueue import SGECluster
import dask

from . import QUEUE_WAIT


@pytest.mark.env("sge")
def test_basic(loop):
    with SGECluster(
        walltime="00:02:00", cores=8, processes=4, memory="2GB", loop=loop
    ) as cluster:
        with Client(cluster, loop=loop) as client:

            cluster.scale(2)

            start = time()
            while not client.scheduler_info()["workers"]:
                sleep(0.100)
                assert time() < start + QUEUE_WAIT

            future = client.submit(lambda x: x + 1, 10)
            assert future.result(QUEUE_WAIT) == 11
            assert len(client.scheduler_info()["workers"]) > 0

            workers = list(client.scheduler_info()["workers"].values())
            w = workers[0]
            assert w["memory_limit"] == 2e9 / 4
            assert w["nthreads"] == 2

            cluster.scale(0)

            start = time()
            while client.scheduler_info()["workers"]:
                sleep(0.100)
                assert time() < start + QUEUE_WAIT


@pytest.mark.env("sge")
def test_basic_ta(loop):
    with SGECluster(
        walltime="00:02:00", cores=1, processes=1, memory="2GB", loop=loop,
        job_extra=["-t 1-2"]
    ) as cluster:
        with Client(cluster, loop=loop) as client:

            cluster.scale(2)

            assert (
                len(cluster.worker_spec) == 2
            ), f"cluster.worker_spec should be 2, {cluster.worker_spec} found"

            start = time()
            while not client.scheduler_info()["workers"]:
                sleep(0.100)
                assert time() < start + QUEUE_WAIT

            start = time()
            while len(client.scheduler_info()["workers"]) < 4:
                sleep(0.100)
                assert time() < start + QUEUE_WAIT
            assert (
                len(cluster.scheduler_info["workers"]) == 4
            ), f"cluster.scheduler_info should have 4 workers, {len(cluster.scheduler_info['workers'])} found"
            expected_names = ['0_1', '0_2', '1_1', '1_2']
            workers_names = sorted(i['id'] for i in cluster.scheduler_info["workers"].values())
            assert workers_names == expected_names, f'cluster.scheduler_info should have the following ids: {expected_names}. {workers_names} found instead'


@pytest.mark.env("sge")
def test_processes_ta(loop):
    with SGECluster(
        walltime="00:02:00", cores=4, processes=2, memory="2GB", loop=loop,
        job_extra=["-t 1-2", "-pe smp 4"]
    ) as cluster:
        with Client(cluster, loop=loop) as client:

            cluster.scale(1)

            assert (
                len(cluster.worker_spec) == 1
            ), f"cluster.worker_spec should be 1, {cluster.worker_spec} found"

            start = time()
            while not client.scheduler_info()["workers"]:
                sleep(0.100)
                assert time() < start + QUEUE_WAIT

            start = time()
            while len(client.scheduler_info()["workers"]) < 4:
                sleep(0.100)
                assert time() < start + QUEUE_WAIT
            assert (
                len(cluster.scheduler_info["workers"]) == 4
            ), f"cluster.scheduler_info should have 4 workers, {len(cluster.scheduler_info['workers'])} found"
            expected_names = ['0_1-0', '0_1-1', '0_2-0', '0_2-1']
            workers_names = sorted(i['id'] for i in cluster.scheduler_info["workers"].values())
            assert workers_names == expected_names, f'cluster.scheduler_info should have the following ids: {expected_names}. {workers_names} found instead'


def test_config_name_sge_takes_custom_config():
    conf = {
        "queue": "myqueue",
        "project": "myproject",
        "ncpus": 1,
        "cores": 1,
        "memory": "2 GB",
        "walltime": "00:02",
        "job-extra": [],
        "name": "myname",
        "processes": 1,
        "interface": None,
        "death-timeout": None,
        "local-directory": "/foo",
        "extra": [],
        "env-extra": [],
        "log-directory": None,
        "shebang": "#!/usr/bin/env bash",
        "job-cpu": None,
        "job-mem": None,
        "resource-spec": None,
    }

    with dask.config.set({"jobqueue.sge-config-name": conf}):
        with SGECluster(config_name="sge-config-name") as cluster:
            assert cluster.job_name == "myname"


def test_job_script(tmpdir):
    log_directory = tmpdir.strpath
    with SGECluster(
        cores=6,
        processes=2,
        memory="12GB",
        queue="my-queue",
        project="my-project",
        walltime="02:00:00",
        env_extra=["export MY_VAR=my_var"],
        job_extra=["-w e", "-m e"],
        log_directory=log_directory,
        resource_spec="h_vmem=12G,mem_req=12G",
    ) as cluster:
        job_script = cluster.job_script()
        for each in [
            "--nprocs 2",
            "--nthreads 3",
            "--memory-limit 6.00GB",
            "-q my-queue",
            "-P my-project",
            "-l h_rt=02:00:00",
            "export MY_VAR=my_var",
            "#$ -w e",
            "#$ -m e",
            "#$ -e {}".format(log_directory),
            "#$ -o {}".format(log_directory),
            "-l h_vmem=12G,mem_req=12G",
            "#$ -cwd",
            "#$ -j y",
        ]:
            assert each in job_script


@pytest.mark.env("sge")
def test_complex_cancel_command(loop):
    with SGECluster(
        walltime="00:02:00", cores=1, processes=1, memory="2GB", loop=loop
    ) as cluster:
        with Client(cluster) as client:
            username = "root"
            cluster.cancel_command = "qdel -u {}".format(username)

            cluster.scale(2)

            start = time()
            while not client.scheduler_info()["workers"]:
                sleep(0.100)
                assert time() < start + QUEUE_WAIT

            cluster.scale(0)

            start = time()
            while client.scheduler_info()["workers"]:
                sleep(0.100)
                assert time() < start + QUEUE_WAIT
