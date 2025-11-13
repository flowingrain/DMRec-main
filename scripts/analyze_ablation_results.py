"""
分析消融实验结果，计算各修改的独立贡献
"""
import re
import json
from pathlib import Path


def parse_log_file(log_path):
    """
    从日志文件中解析实验结果
    
    Args:
        log_path: 日志文件路径
    
    Returns:
        dict: 包含各项指标的结果字典
    """
    with open(log_path, 'r') as f:
        content = f.read()
    
    # 匹配结果行（根据实际日志格式调整）
    # 示例格式: "Final test result: {'recall': array([0.1048, 0.1530]), 'ndcg': array([0.0793, 0.0952])}"
    pattern = r"Final test result: \{'recall': array\(\[([\d.]+),\s*([\d.]+)\]\),\s*'ndcg': array\(\[([\d.]+),\s*([\d.]+)\]\)\}"
    match = re.search(pattern, content)
    
    if match:
        return {
            'recall_10': float(match.group(1)),
            'recall_20': float(match.group(2)),
            'ndcg_10': float(match.group(3)),
            'ndcg_20': float(match.group(4))
        }
    
    # 如果没有匹配到，尝试其他格式
    return None


def extract_best_epoch(log_path):
    """提取最佳 epoch"""
    with open(log_path, 'r') as f:
        content = f.read()
    
    pattern = r"Best Epoch (\d+)"
    match = re.search(pattern, content)
    
    if match:
        return int(match.group(1))
    return None


def calculate_contributions(results):
    """
    计算各修改的独立贡献
    
    Args:
        results: dict, 包含所有实验结果
    """
    baseline = results.get('baseline', {})
    
    contributions = {}
    
    # 独立贡献
    if 'reflow_only' in results:
        contributions['reflow'] = {
            'recall_10': results['reflow_only']['recall_10'] - baseline['recall_10'],
            'recall_20': results['reflow_only']['recall_20'] - baseline['recall_20'],
            'ndcg_10': results['reflow_only']['ndcg_10'] - baseline['ndcg_10'],
            'ndcg_20': results['reflow_only']['ndcg_20'] - baseline['ndcg_20'],
        }
    
    if 'adaptive_only' in results:
        contributions['adaptive'] = {
            'recall_10': results['adaptive_only']['recall_10'] - baseline['recall_10'],
            'recall_20': results['adaptive_only']['recall_20'] - baseline['recall_20'],
            'ndcg_10': results['adaptive_only']['ndcg_10'] - baseline['ndcg_10'],
            'ndcg_20': results['adaptive_only']['ndcg_20'] - baseline['ndcg_20'],
        }
    
    if 'vectorfield_only' in results:
        contributions['vectorfield'] = {
            'recall_10': results['vectorfield_only']['recall_10'] - baseline['recall_10'],
            'recall_20': results['vectorfield_only']['recall_20'] - baseline['recall_20'],
            'ndcg_10': results['vectorfield_only']['ndcg_10'] - baseline['ndcg_10'],
            'ndcg_20': results['vectorfield_only']['ndcg_20'] - baseline['ndcg_20'],
        }
    
    # 交互效应（如果有组合实验）
    if 'reflow_adaptive' in results and 'reflow_only' in results and 'adaptive_only' in results:
        contributions['interaction_reflow_adaptive'] = {
            'recall_10': (results['reflow_adaptive']['recall_10'] - baseline['recall_10']) - 
                        (contributions['reflow']['recall_10'] + contributions['adaptive']['recall_10']),
            'recall_20': (results['reflow_adaptive']['recall_20'] - baseline['recall_20']) - 
                        (contributions['reflow']['recall_20'] + contributions['adaptive']['recall_20']),
            'ndcg_10': (results['reflow_adaptive']['ndcg_10'] - baseline['ndcg_10']) - 
                       (contributions['reflow']['ndcg_10'] + contributions['adaptive']['ndcg_10']),
            'ndcg_20': (results['reflow_adaptive']['ndcg_20'] - baseline['ndcg_20']) - 
                       (contributions['reflow']['ndcg_20'] + contributions['adaptive']['ndcg_20']),
        }
    
    return contributions


def generate_report(results, contributions):
    """生成分析报告"""
    report = []
    report.append("# RFDM 消融实验结果分析\n")
    report.append(f"生成时间: {Path(__file__).stat().st_mtime}\n")
    
    # 结果对比表
    report.append("## 实验结果对比\n")
    report.append("| 实验 | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 | vs Baseline |")
    report.append("|------|-----------|-----------|---------|---------|-------------|")
    
    baseline = results.get('baseline', {})
    for exp_name, exp_results in results.items():
        if exp_name == 'baseline':
            vs_baseline = "-"
        else:
            diff = exp_results['recall_10'] - baseline['recall_10']
            vs_baseline = f"{diff:+.4f} ({diff/baseline['recall_10']*100:+.2f}%)"
        
        report.append(
            f"| {exp_name} | {exp_results['recall_10']:.6f} | {exp_results['recall_20']:.6f} | "
            f"{exp_results['ndcg_10']:.6f} | {exp_results['ndcg_20']:.6f} | {vs_baseline} |"
        )
    
    # 独立贡献分析
    report.append("\n## 独立贡献分析\n")
    report.append("| 修改 | Recall@10 贡献 | Recall@20 贡献 | NDCG@10 贡献 | NDCG@20 贡献 |")
    report.append("|------|---------------|---------------|--------------|--------------|")
    
    for mod_name, contrib in contributions.items():
        if 'interaction' not in mod_name:
            report.append(
                f"| {mod_name} | {contrib['recall_10']:+.6f} | {contrib['recall_20']:+.6f} | "
                f"{contrib['ndcg_10']:+.6f} | {contrib['ndcg_20']:+.6f} |"
            )
    
    # 交互效应
    if any('interaction' in k for k in contributions.keys()):
        report.append("\n## 交互效应分析\n")
        report.append("| 组合 | Recall@10 交互 | Recall@20 交互 | NDCG@10 交互 | NDCG@20 交互 |")
        report.append("|------|---------------|---------------|--------------|--------------|")
        
        for mod_name, contrib in contributions.items():
            if 'interaction' in mod_name:
                report.append(
                    f"| {mod_name} | {contrib['recall_10']:+.6f} | {contrib['recall_20']:+.6f} | "
                    f"{contrib['ndcg_10']:+.6f} | {contrib['ndcg_20']:+.6f} |"
                )
    
    # 结论
    report.append("\n## 结论\n")
    
    # 找出贡献最大的修改
    if contributions:
        best_contrib = max(contributions.items(), 
                         key=lambda x: x[1]['recall_10'] if 'interaction' not in x[0] else -1)
        report.append(f"- **最大独立贡献**: {best_contrib[0]} (Recall@10: {best_contrib[1]['recall_10']:+.6f})")
    
    return "\n".join(report)


def main():
    """主函数"""
    log_dir = Path("logs")
    results = {}
    
    # 解析所有日志文件
    experiments = ['baseline', 'reflow_only', 'adaptive_only', 'vectorfield_only', 
                   'reflow_adaptive', 'all']
    
    for exp_name in experiments:
        log_path = log_dir / f"{exp_name}.log"
        if log_path.exists():
            result = parse_log_file(log_path)
            if result:
                results[exp_name] = result
                print(f"Parsed {exp_name}: {result}")
            else:
                print(f"Warning: Could not parse {exp_name}")
        else:
            print(f"Warning: Log file not found for {exp_name}")
    
    if not results:
        print("No results found! Please run experiments first.")
        return
    
    # 计算贡献
    contributions = calculate_contributions(results)
    
    # 生成报告
    report = generate_report(results, contributions)
    
    # 保存报告
    output_path = Path("ablation_analysis_report.md")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nAnalysis report saved to {output_path}")
    print("\n" + report)


if __name__ == "__main__":
    main()

