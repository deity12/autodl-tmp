如何编译 `thesis_proposal.tex` 为 PDF

先决条件（Windows）:
- 安装 TeX 发行版：MikTeX 或 TeX Live，并安装支持 XeLaTeX 的组件。
- 或使用在线 Overleaf 编译（上传 .tex 文件）。

推荐使用 XeLaTeX（支持中文）：
在项目目录（包含 `thesis_proposal.tex`）打开 PowerShell 或 CMD，运行：

```powershell
xelatex -interaction=nonstopmode thesis_proposal.tex
xelatex -interaction=nonstopmode thesis_proposal.tex
```

说明：两次编译可确保目录与引用正确。若使用 bibtex/biber，请根据需要运行相应命令。

常见问题：
- 若缺少中文字体或 ctex 宏包，确保 TeX Live 或 MikTeX 已安装 ctex 包（MikTeX 会在首次编译时自动安装所需包）。
- Windows/MikTeX 用户：可用 MikTeX Console 更新包，或直接在 TeX Live Utility 中安装。

我可以：
- 帮您在本地尝试编译（若您允许我在此环境执行 XeLaTeX）。
- 或将生成的 PDF 通过工作区返回（需我先在环境中编译）。

下一步建议：如果要我帮忙编译，请回复“请编译PDF”。
