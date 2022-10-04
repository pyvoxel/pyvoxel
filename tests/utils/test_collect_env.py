from voxel.utils.collect_env import collect_env_info


def test_collect_env_info():
    env_info = collect_env_info()
    assert isinstance(env_info, str)
