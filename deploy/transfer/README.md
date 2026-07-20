# 完整包还原

本目录支持两种交付形式：单个 `*.tar.gz` 加 `archive.sha256`，或者所有
`*.tar.gz.part-*` 分卷加 `parts.sha256`。请同时保留还原脚本。

Windows：

```powershell
.\restore.cmd D:\maiagent-training
```

Linux/macOS：

```bash
chmod +x restore.sh
./restore.sh /data/maiagent-training
```

脚本会先校验 SHA-256，再还原完整目录。只有分卷模式在 Windows 上需要额外临时空间
拼接压缩包；单文件模式会直接解压。还原完成后进入生成的
`maiagent-muq-audio-complete-*` 目录，按其中 `README.md` 执行。
