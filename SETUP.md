# Short Drama Replicator - 环境配置

1. 在 GitHub 创建私有仓库 `short-drama-tasks`
2. 创建 Personal Access Token (Settings → Developer settings → Personal access tokens → Tokens (classic))
   - Scope: `repo`
3. 复制 `.env.example` 为 `.env`，填入实际值
4. 安装依赖: `pip install -r requirements.txt`
5. 确保 ffmpeg 在 PATH 中
6. 确保 videocaptioner 可用: `~/.local/share/short-drama-automation/vc-venv/bin/videocaptioner`

## GitHub 中转仓库初始化

```bash
mkdir -p /tmp/short-drama-tasks/tasks/{pending,running,done,failed}
cd /tmp/short-drama-tasks
for d in pending running done failed; do
  echo "" > tasks/$d/.gitkeep
done
git init
git add .
git commit -m "init task repo"
git remote add origin https://github.com/YOUR_USERNAME/short-drama-tasks.git
git push -u origin main
```
