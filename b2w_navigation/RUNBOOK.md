# 当前默认运行流程

使用 [WRIST_SPORT_APPROACH.md](WRIST_SPORT_APPROACH.md)：Sport桥+里程计，腕部搜索确认后通过Move速度闭环到海绵左/右0.5m。默认接近不依赖SLAM/Nav2，尚未真机验收。

只搜索见 [WRIST_SEARCH.md](WRIST_SEARCH.md)。需要原Nav2探索/接近时见 [WRIST_APPROACH.md](WRIST_APPROACH.md)，并明确指定 `--approach-backend nav2`。

旧的底盘单独导航说明在 [RUNBOOK_BASE_ONLY.md](RUNBOOK_BASE_ONLY.md)。
