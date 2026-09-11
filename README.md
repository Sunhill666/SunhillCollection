# 日山集 · Sunhill Collection

个人技术花园。这里整理读过的论文、解决过的工程问题，以及仍在推演中的系统设计。

🔗 https://garden.instanext.cn

## 内容结构

| 目录                      | 内容                                                   |
| ------------------------- | ------------------------------------------------------ |
| `content/Paper Reading/`  | 论文阅读笔记：目标检测、跟踪、低标注学习、视觉语言模型 |
| `content/Learning/`       | 学习札记：概念梳理与总结                               |
| `content/Technical Blog/` | 工程实践：CUDA、Kubernetes、可观测性                   |

笔记为 Obsidian 风格的 Markdown，frontmatter 使用 `title` / `aliases` / `created` / `tags`。

## 本地开发

```bash
npm install
npx quartz build --serve   # http://localhost:8080
```

其他命令：

```bash
npm run check    # 类型检查 + 格式检查
npm run format   # 格式化
npm test         # 运行测试
```

## 部署

Vercel 自动构建：

- Framework Preset: `Other`
- Build Command: `npx quartz plugin install && npx quartz build`
- Output Directory: `public`

## 内容授权

笔记与文章的文字内容采用 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 许可：转载请注明出处（日山集 · Sunhill Collection 及原文链接），禁止商业使用，改编后需以相同方式共享。

文中引用的论文图表与摘要原文，版权归原作者及出版方所有。

## 说明

基于 [Quartz v5](https://quartz.jzhao.xyz/) 构建，代码部分遵循 MIT 许可（见 `LICENSE`）。主题来自 [quartz-themes](https://github.com/quartz-themes)，插件来自 [quartz-community](https://github.com/quartz-community)。
