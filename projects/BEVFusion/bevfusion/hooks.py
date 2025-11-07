# projects/BEVFusion/bevfusion/hooks.py
import os
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

# 네가 작성한 PedObjectSample 경로 맞춰서 import
from projects.BEVFusion.bevfusion.transforms_3d import PedObjectSample, CutAndPaste

@HOOKS.register_module()
class SetPedObjectSampleEpochHook(Hook):
    """매 epoch 시작 전에 PedObjectSample.set_epoch(epoch) 호출"""

    def before_train_epoch(self, runner):
        # 1-based epoch로 저장하고 싶으면 +1, 0-based면 그대로 사용
        epoch = runner.epoch + 1

        # dataloader에서 실제 dataset 꺼내기 (Wrapper 여러겹 방지)
        dataset = runner.train_dataloader.dataset
        while hasattr(dataset, 'dataset'):
            dataset = dataset.dataset

        # pipeline 얻기 (Compose)
        pipeline = getattr(dataset, 'pipeline', None)
        if pipeline is None:
            return

        transforms = getattr(pipeline, 'transforms', pipeline)

        # 여러 개 있을 수도 있으니 전부 순회하며 세팅
        for t in transforms:
            if isinstance(t, PedObjectSample) or isinstance(t, CutAndPaste):
                t.set_epoch(epoch)  # log_path 갱신
