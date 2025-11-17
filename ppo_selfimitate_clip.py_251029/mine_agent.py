from __future__ import annotations

import torch.nn as nn
from typing import Optional  # --- DPO 변경: Optional 임포트 ---
from .batch import Batch


# my implementation 9-7, actor and critic don't share parameters
class MineAgent(nn.Module):
    def __init__(
        self,
        actor: nn.Module,
        critic: Optional[nn.Module] = None, # --- DPO 변경: critic이 None일 수 있도록 수정 ---
        deterministic_eval: bool = False, # use stochastic in both exploration and test
    ):
        super().__init__()
        self.actor = actor
        self.critic = critic # critic이 None으로 저장될 수 있음
        self._deterministic_eval = deterministic_eval
        self.dist_fn = actor.dist_fn

    # forward actor (수정 필요 없음)
    def forward(
        self,
        batch: Batch,
        state=None,
        **kwargs,
    ) -> Batch:
        logits, hidden = self.actor(batch.obs, state=state)
        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)
        if self._deterministic_eval and not self.training:
            act = dist.mode()
        else:
            act = dist.sample()
        return Batch(logits=logits, act=act, state=hidden, dist=dist)

    '''
    # input an obs, output the action distribution
    def _distribution(self, obs):
        logits, _ = self.actor(batch.obs, state=state)
        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)
        return dist
    '''

    # forward actor critic
    def forward_actor_critic(
        self,
        batch: Batch
    ) -> Batch:
        logits, _ = self.actor(batch.obs)
        
        # --- DPO 변경: self.critic이 None일 경우 val 계산 건너뛰기 ---
        val = None
        if self.critic is not None:
            val = self.critic(batch.obs)
        # --- DPO 변경 끝 ---
        
        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)
        if self._deterministic_eval and not self.training:
            act = dist.mode()
        else:
            act = dist.sample()
        logp = dist.log_prob(act)

        # val이 None일 수 있음 (DPO에서는 val을 사용하지 않으므로 괜찮음)
        return Batch(logits=logits, act=act, dist=dist, logp=logp, val=val)
