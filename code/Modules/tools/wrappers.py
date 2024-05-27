# -*- coding: utf-8 -*-
# @Time : 2022/10/16 14:01
# @Author : Mengtian Zhang
# @E-mail : zhangmengtian@sjtu.edu.cn
# @Version : v-dev-0.0
# @License : MIT License
# @Copyright : Copyright 2022, Mengtian Zhang
# @Function 

"""Summary.

Description.----------------------------------------------------------------
----------------------------------------------------------------------------
----------------------------------------------------------------------------

Example:

"""


class SchedulerWrapper:
    def __init__(self, scheduler_list):
        self.scheduler_list = scheduler_list

    def step(self, *args, **kwargs):
        for scheduler in self.scheduler_list:
            scheduler.step(*args, **kwargs)
