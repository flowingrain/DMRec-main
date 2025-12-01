# 不修改 acmart.cls 的字体修复方案

## 概述

本方案使用 **acmart-preload-hook.tex** 机制来禁用 Libertine/NewTX 字体，**无需修改 acmart.cls 文件**。

## 工作原理

### 1. ACM 模板的 Preload Hook 机制

ACM 模板 (`acmart.cls`) 在第43-46行支持一个 preload hook：

```latex
\InputIfFileExists{acmart-preload-hook.tex}{%
  \ClassWarning{\@classname}{%
    I am loading acmart-preload-hook.tex. You are fully responsible
    for any problems from now on.}}{}
```

如果 `acmart-preload-hook.tex` 文件存在于同一目录，它会在类文件处理选项**之前**被加载。

### 2. 拦截字体包加载

`acmart-preload-hook.tex` 的工作原理：

1. **保存原始的 `\IfFileExists` 命令**
2. **重新定义 `\IfFileExists`**，使其对以下字体包返回"不存在"：
   - `libertine.sty`
   - `newtxmath.sty`
   - `zi4.sty` (Inconsolata)
3. **acmart.cls 检查文件存在性时**，会认为这些包不存在
4. **acmart 自动设置 `\@ACM@newfontsfalse`**，从而不加载这些字体包
5. **类文件加载完成后**，恢复原始的 `\IfFileExists` 命令

### 3. 字体设置覆盖

在 `ReFlowRec.tex` 中，我们在 `\documentclass` 之后：

1. 设置 `\@ACM@newfontsfalse`（作为备份）
2. 强制使用 Computer Modern 字体
3. 重新定义所有数学字体族

## 文件结构

```
ReFlowRec/
├── acmart.cls                    # 原始文件，未修改
├── acmart-preload-hook.tex       # 新增：拦截字体包加载
└── ReFlowRec.tex                 # 主文档，包含字体覆盖代码
```

## 优势

✅ **不修改模板文件**：`acmart.cls` 保持原始状态  
✅ **易于维护**：如果更新 acmart.cls，只需保留 preload hook 文件  
✅ **符合 ACM 设计**：使用官方支持的 hook 机制  
✅ **可移植**：只需复制 `acmart-preload-hook.tex` 到其他项目

## 与直接修改 acmart.cls 的对比

| 方法 | 优点 | 缺点 |
|------|------|------|
| **Preload Hook** | 不修改模板，易于维护 | 稍微复杂一些 |
| **直接修改** | 简单直接 | 更新模板时会丢失修改 |

## 关于 Auto Expansion

**问题**：microtype 包默认启用 font expansion，但 `wasysym` 包使用 bitmap 字体（非可缩放），不支持 expansion。

**解决方案**：在 `ReFlowRec.tex` 第18行：
```latex
\PassOptionsToPackage{expansion=false}{microtype}
```

这会禁用所有字体的 auto expansion，避免与 wasysym 冲突。

## 验证

编译文档后，检查 `ReFlowRec.log`：

✅ **应该看到**：
- `Package: libertine` - **不应该出现**
- `Package: newtxmath` - **不应该出现**
- `\rmdefault=cmr` - 使用 Computer Modern

❌ **不应该看到**：
- `Font LibertineMathBRM not found` 错误
- `miktex-makemf` 或 `miktex-makepk` 错误

## 故障排除

### 如果字体包仍然被加载

1. 检查 `acmart-preload-hook.tex` 是否在正确位置（与 `acmart.cls` 同一目录）
2. 检查文件名拼写是否正确
3. 查看 `ReFlowRec.log` 中是否有 hook 加载的警告信息

### 如果仍有字体错误

在 `ReFlowRec.tex` 的 `\documentclass` 之后添加更强制性的覆盖：

```latex
\makeatletter
% 强制卸载已加载的字体包（如果它们被加载了）
\@ifpackageloaded{libertine}{\RequirePackage{libertine}\let\libertine\relax}{}
\@ifpackageloaded{newtxmath}{\RequirePackage{newtxmath}\let\newtxmath\relax}{}
\@ACM@newfontsfalse
\makeatother
```

## 参考

- ACM 模板文档：https://www.acm.org/publications/proceedings-template
- LaTeX `\IfFileExists` 命令文档
- microtype 包文档：https://ctan.org/pkg/microtype

