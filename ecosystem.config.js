module.exports = {
  apps: [
    {
      name: "ai-server",
      script: "/home/pod/folder/zaq/ai-env/bin/python",
      args: "-m uvicorn main:app --host 0.0.0.0 --port 8000",
      cwd: "/home/pod/folder/zaq",
      interpreter: "none",
      autorestart: true,
      watch: false,
      env: {
        PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:True"
      }
    }
  ]
};