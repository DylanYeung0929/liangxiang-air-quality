"""
论文全面检查清单
检查论文的完整性、一致性、格式规范等
"""
from pathlib import Path
import re

project_root = Path(__file__).resolve().parents[2]
tex_file = project_root / "paper/main.tex"

print("=" * 80)
print("论文全面检查")
print("=" * 80)

with open(tex_file, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. 检查摘要是否需要更新
print("\n[1] 摘要检查")
abstract_match = re.search(r'\\noindent\{\\bfseries 摘要：\}(.*?)\\vspace', content, re.DOTALL)
if abstract_match:
    abstract = abstract_match.group(1)
    if 'VECM' in abstract or '稀疏' in abstract or 'Sparse' in abstract:
        print("  [OK] 摘要已包含新模型内容")
    else:
        print("  [!] 摘要未提及VECM和稀疏VAR，建议更新")
        print("      当前摘要提到的模型：")
        if 'VAR' in abstract:
            print("      - VAR")
        if 'Ridge' in abstract:
            print("      - Ridge")

# 2. 检查关键词
print("\n[2] 关键词检查")
keywords_match = re.search(r'\\noindent\{\\bfseries 关键词：\}(.*?)\\clearpage', content, re.DOTALL)
if keywords_match:
    keywords = keywords_match.group(1).strip()
    print(f"  当前关键词: {keywords}")
    if 'VECM' not in keywords and '稀疏' not in keywords:
        print("  [!] 建议添加：稀疏VAR、协整检验")

# 3. 检查图表引用
print("\n[3] 图表引用检查")
# 查找所有\fig和\label
figs = re.findall(r'\\fig\{([^}]+)\}', content)
labels = re.findall(r'\\label\{(fig:[^}]+)\}', content)
refs = re.findall(r'\\ref\{(fig:[^}]+)\}', content)

print(f"  定义的图表数量: {len(figs)}")
print(f"  图表标签数量: {len(labels)}")
print(f"  图表引用数量: {len(refs)}")

# 检查fig20和fig21是否被引用
if 'fig20_sparse_var_coefficients.pdf' in content:
    if 'fig:sparse_coef' in refs:
        print("  [OK] fig20已被引用")
    else:
        print("  [!] fig20未被引用")

if 'fig21_three_model_comparison.pdf' in content:
    if 'fig:three_model' in refs:
        print("  [OK] fig21已被引用")
    else:
        print("  [!] fig21未被引用")

# 4. 检查表格编号
print("\n[4] 表格编号检查")
tables = re.findall(r'\\caption\{([^}]+)\}\\s*\\label\{(tab:[^}]+)\}', content)
print(f"  表格数量: {len(tables)}")

# 检查新增表格
new_tables = ['johansen', 'sparse_var_comparison', 'sparse_var_detail']
for tab in new_tables:
    if f'tab:{tab}' in content:
        print(f"  [OK] 表格 tab:{tab} 已定义")
    else:
        print(f"  [!] 表格 tab:{tab} 未找到")

# 5. 检查公式编号
print("\n[5] 公式编号检查")
equations = re.findall(r'\\label\{(eq:[^}]+)\}', content)
print(f"  公式数量: {len(equations)}")

# 检查新增公式
if 'eq:vecm_var_equiv' in equations:
    print("  [OK] VECM等价性公式已定义")
if 'eq:sparse_var' in equations:
    print("  [OK] 稀疏VAR公式已定义")

# 6. 检查章节结构
print("\n[6] 章节结构检查")
sections = re.findall(r'\\(sub)*section\{([^}]+)\}', content)
print(f"  章节总数: {len(sections)}")

# 检查是否有"进一步改进尝试"章节
if '进一步改进尝试' in content or 'VECM 与稀疏 VAR' in content:
    print("  [OK] 新增章节已添加")
else:
    print("  [!] 新增章节未找到")

# 7. 检查文献引用
print("\n[7] 文献引用检查")
citations = re.findall(r'\\cite\{([^}]+)\}', content)
print(f"  文献引用次数: {len(citations)}")
if len(citations) > 0:
    print(f"  引用的文献: {set(citations)}")

# 8. 检查附录代码
print("\n[8] 附录代码检查")
lstinputlisting = re.findall(r'\\lstinputlisting.*?\{([^}]+)\}', content)
print(f"  代码文件数量: {len(lstinputlisting)}")
for code_file in lstinputlisting:
    print(f"    - {code_file}")

# 9. 检查AI使用报告
print("\n[9] AI使用报告检查")
if 'AI使用情况报告' in content:
    print("  [OK] AI使用报告已包含")
    # 检查是否提到新实验
    ai_section = content[content.find('AI使用情况报告'):content.find('\\section{附录}')]
    if 'VECM' in ai_section or '稀疏' in ai_section:
        print("  [!] AI报告可能需要更新，提及新实验")
    else:
        print("  [!] AI报告未提及VECM和稀疏VAR实验")

# 10. 检查页数和文件大小
print("\n[10] 文档规模检查")
pdf_file = project_root / "paper/main.pdf"
if pdf_file.exists():
    size_mb = pdf_file.stat().st_size / (1024 * 1024)
    print(f"  PDF大小: {size_mb:.2f} MB")
    if size_mb > 10:
        print("  [!] 文件过大，可能需要压缩图片")
    else:
        print("  [OK] 文件大小合理")

print("\n" + "=" * 80)
print("检查完成")
print("=" * 80)
