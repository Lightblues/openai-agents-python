NOTE: 参见 [agent_ref/doc]

## 更新到 0.1.13
1. 支持 LiteLLM 模型 (src/agents/extensions/models/litellm_model.py)
2. 支持 multi_provider (src/agents/models/multi_provider.py)
3. 支持 extra_headers 参数 (src/agents/model_settings.py)
4. 支持非严格JSON输出 (src/agents/agent_output.py)
5. 分离 Converter (src/agents/models/chatcmpl_converter.py)
6. 分离 ChatCmlpStreamHandler (src/agents/models/chatcmpl_stream_handler.py)
7. 对于 RunResultStreaming 增加 `cancel` 方法 (src/agents/result.py)

## RULES
1. 透明封装? 
2. 变更: 将修改点记录到 OVERWRITE.md

## Misc
[SDK包改造方案]
├── 方案对比
│   ├── Fork+分支方案（直接修改）
│   │   ✓ 适合简单直接修改
│   │   ✗ 需处理上游合并冲突
│   ├── 透明封装方案
│   │   ✓ 完全隔离底层变化
│   │   ✗ 需设计良好接口
│   └── Git子模块方案
│       ✓ 保持原仓库独立性
│       ✗ 需管理子模块版本
└── 推荐方案
采用Fork仓库 + Git子模块组合：
1. Fork原仓库到自有Git账户
2. 创建dev分支进行定制开发
3. 通过git submodule将SDK引入应用仓库 (pip install -e ./lib/openai-agents)
4. 定期执行上游同步：
```sh
git remote add upstream https://github.com/openai/openai-agents-python
git fetch upstream
git merge upstream/main
```

[建议]
- 变更管理
    - 维护OVERWRITE.md记录所有修改
    - 使用diff工具定期对比原仓库： `git diff upstream/main..feature/custom`
- 高级技巧
    - 使用git worktree同时维护多个分支
    - 通过pre-commit hook防止意外提交到错误分支
    - 利用GitHub Actions实现自动同步检测：
        - 定时检查上游更新
        - 自动创建PR同步请求
        - 触发回归测试流水线