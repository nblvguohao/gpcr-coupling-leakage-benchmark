#!/usr/bin/env python3
"""
批量运行 TMHMM2 / Phobius 拓扑预测。

逻辑:
1. 读取 merged_dataset/extended_sequences.json
2. 优先尝试本地 tmhmm 命令行; 如未找到则写 FASTA 文件并提供在线提交指引
3. 解析 TMHMM2 short 输出格式，提取 TM1-7 和 loop 区域
4. 输出 JSON: {uid: {tm_regions: [(s,e), ...], loops: {ICL1/ECL1/...: (s,e)}}}
"""

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).parent
SEQUENCES_FILE = BASE / "merged_dataset" / "extended_sequences.json"
OUTPUT_DIR = BASE / "paired_dataset"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "tmhmm_topology.json"
FASTA_OUTPUT = BASE / "tmhmm_results" / "batch_submission.fasta"


def sequences_to_fasta(sequences: dict, out_path: Path):
    """将序列字典写成 FASTA。"""
    with open(out_path, "w") as f:
        for uid, record in sequences.items():
            seq = record["sequence"] if isinstance(record, dict) else record
            f.write(f">{uid}\n{seq}\n")


def find_tmhmm():
    """查找本地 tmhmm 可执行文件。"""
    return shutil.which("tmhmm") or shutil.which("tmhmm.py") or shutil.which("tmhmm2")


def run_tmhmm_local(fasta_path: Path, out_dir: Path):
    """调用本地 tmhmm。"""
    tmhmm_bin = find_tmhmm()
    if not tmhmm_bin:
        return None
    cmd = [tmhmm_bin, str(fasta_path), "-WorkDir", str(out_dir)]
    # 某些版本的 tmhmm.py 参数格式不同，这里尝试通用写法
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return out_dir
    except subprocess.CalledProcessError:
        # 尝试 tmhmm2 / Decoy 版本参数
        cmd2 = [tmhmm_bin, "--workdir", str(out_dir), str(fasta_path)]
        try:
            subprocess.run(cmd2, check=True, capture_output=True, text=True)
            return out_dir
        except subprocess.CalledProcessError:
            return None


def parse_tmhmm_short(short_output: str):
    """
    解析 TMHMM2 short 输出。
    示例行:
    NET_MYGS_HUMAN        TMHMM2.0    inside     1    34
    NET_MYGS_HUMAN        TMHMM2.0    TMhelix   35    57
    NET_MYGS_HUMAN        TMHMM2.0    outside   58    68
    """
    lines = [l.strip() for l in short_output.splitlines() if l.strip()]
    regions = []
    for line in lines:
        parts = re.split(r"\s+", line)
        if len(parts) >= 5 and parts[1].startswith("TMHMM"):
            region_type = parts[2]
            start = int(parts[3])
            end = int(parts[4])
            regions.append((region_type, start, end))
    return regions


def tm_regions_from_parsed(regions):
    """从 parsed regions 中提取 TM1-7 并推导 loop 区域。"""
    tm_regions = []
    for rtype, s, e in regions:
        if rtype == "TMhelix":
            tm_regions.append((s, e))

    # 最多保留 7 个 TMhelix
    if len(tm_regions) > 7:
        # 如果 >7，按长度保留最长的 7 个 (GPCR 通常为 7次跨膜)
        tm_regions = sorted(tm_regions, key=lambda x: x[1] - x[0], reverse=True)[:7]
        # 再按起始位置排序
        tm_regions = sorted(tm_regions, key=lambda x: x[0])

    loops = {}
    if not tm_regions:
        return tm_regions, loops

    # N-tail
    n_tail_end = tm_regions[0][0] - 1
    if n_tail_end >= 1:
        loops["N-tail"] = (1, n_tail_end)

    # TM 之间的 loops
    loop_names = ["ICL1", "ECL1", "ICL2", "ECL2", "ICL3", "ECL3"]
    for i in range(len(tm_regions) - 1):
        gap_s = tm_regions[i][1] + 1
        gap_e = tm_regions[i + 1][0] - 1
        if gap_e >= gap_s:
            name = loop_names[i] if i < len(loop_names) else f"loop_{i}"
            loops[name] = (gap_s, gap_e)

    # C-tail
    c_tail_start = tm_regions[-1][1] + 1
    loops["C-tail"] = (c_tail_start, 999999)  # 后续用序列长度截断

    return tm_regions, loops


def process_local_tmhmm_results(tmhmm_out_dir: Path, sequence_length_dict: dict):
    """扫描 tmhmm 输出目录并解析所有 short 文件。"""
    results = {}
    short_files = list(tmhmm_out_dir.glob("*.short"))
    if not short_files:
        # 某些版本输出到 summary 或不同扩展名
        short_files = list(tmhmm_out_dir.glob("*.summary"))

    for sf in short_files:
        uid = sf.stem
        text = sf.read_text()
        parsed = parse_tmhmm_short(text)
        tm_regions, loops = tm_regions_from_parsed(parsed)
        # 截断 C-tail
        if "C-tail" in loops and uid in sequence_length_dict:
            loops["C-tail"] = (loops["C-tail"][0], sequence_length_dict[uid])
        results[uid] = {
            "tm_regions": tm_regions,
            "loops": loops,
        }
    return results


def run_phobius_web_batch(fasta_path: Path):
    """
    Phobius 在线批量提交说明。
    (此处仅打印指引，不实际请求，因为在线服务有 rate-limit / 需要表单提交)
    """
    print("\n" + "=" * 70)
    print("  Phobius 在线批量提交指引")
    print("=" * 70)
    print(f"FASTA 文件已生成: {fasta_path}")
    print("请访问 http://phobius.sbc.su.se/ 并上传该 FASTA 文件。")
    print("下载结果后，使用 --parse-phobius 参数运行本脚本进行二次解析。\n")


def parse_phobius_short(phobius_text: str, uid: str, seq_len: int):
    """
    解析 Phobius short 输出 (与 TMHMM 格式类似)。
    示例:
    ID   sp|P41595|5HT2B_HUMAN
    FT   TRANSMEM     40    62
    FT   TRANSMEM     78   100
    """
    tm_regions = []
    for line in phobius_text.splitlines():
        if line.startswith("FT   TRANSMEM"):
            parts = line.split()
            if len(parts) >= 4:
                s = int(parts[2])
                e = int(parts[3])
                tm_regions.append((s, e))

    # 与 TMHMM 同样的后处理
    if len(tm_regions) > 7:
        tm_regions = sorted(tm_regions, key=lambda x: x[1] - x[0], reverse=True)[:7]
        tm_regions = sorted(tm_regions, key=lambda x: x[0])

    loops = {}
    if tm_regions:
        if tm_regions[0][0] > 1:
            loops["N-tail"] = (1, tm_regions[0][0] - 1)
        loop_names = ["ICL1", "ECL1", "ICL2", "ECL2", "ICL3", "ECL3"]
        for i in range(len(tm_regions) - 1):
            s = tm_regions[i][1] + 1
            e = tm_regions[i + 1][0] - 1
            if e >= s:
                loops[loop_names[i] if i < len(loop_names) else f"loop_{i}"] = (s, e)
        loops["C-tail"] = (tm_regions[-1][1] + 1, seq_len)

    return {"tm_regions": tm_regions, "loops": loops}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="批量 TMHMM2 / Phobius 拓扑预测")
    parser.add_argument("--parse-phobius", type=Path, default=None,
                        help="解析已下载的 Phobius 结果文件或目录")
    args = parser.parse_args()

    with open(SEQUENCES_FILE) as f:
        sequences = json.load(f)

    seq_lengths = {}
    for uid, rec in sequences.items():
        if isinstance(rec, dict):
            seq_lengths[uid] = rec.get("length", len(rec.get("sequence", "")))
        else:
            seq_lengths[uid] = len(rec)

    # ── 模式 A: 解析已下载的 Phobius 结果 ──
    if args.parse_phobius:
        phobius_path = args.parse_phobius
        results = {}
        if phobius_path.is_dir():
            files = sorted(phobius_path.glob("*"))
            for pf in files:
                uid = pf.stem
                text = pf.read_text()
                results[uid] = parse_phobius_short(text, uid, seq_lengths.get(uid, 99999))
        else:
            # 假设是合并的短输出文件
            text = phobius_path.read_text()
            # Phobius 批量输出通常按 // 或空行分隔
            # 简单处理: 把整个文件当作一个样本 (如果只有一条)
            # 更复杂的情况建议用户按样本拆分成多个文件
            uid = phobius_path.stem
            results[uid] = parse_phobius_short(text, uid, seq_lengths.get(uid, 99999))

        with open(OUTPUT_FILE, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[OK] Phobius 解析完成，结果保存至 {OUTPUT_FILE}")
        return

    # ── 模式 B: 本地 TMHMM2 ──
    tmhmm_bin = find_tmhmm()
    if tmhmm_bin:
        print(f"[INFO] 发现本地 TMHMM2: {tmhmm_bin}")
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta_path = Path(tmpdir) / "batch.fasta"
            sequences_to_fasta(sequences, fasta_path)
            out_dir = Path(tmpdir) / "tmhmm_out"
            out_dir.mkdir(exist_ok=True)
            success_dir = run_tmhmm_local(fasta_path, out_dir)
            if success_dir:
                results = process_local_tmhmm_results(success_dir, seq_lengths)
                with open(OUTPUT_FILE, "w") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"[OK] 本地 TMHMM2 解析完成: {len(results)} 条序列")
                print(f"    结果保存至 {OUTPUT_FILE}")
                return
            else:
                print("[WARN] 本地 TMHMM2 运行失败， fallback 到 FASTA 生成。")

    # ── 模式 C: 生成 FASTA，提示用户在线提交 ──
    FASTA_OUTPUT.parent.mkdir(exist_ok=True)
    sequences_to_fasta(sequences, FASTA_OUTPUT)
    run_phobius_web_batch(FASTA_OUTPUT)

    # 同时生成 TMHMM 在线版提交文件 (TMHMM 在线版不支持批量，只能分条)
    # 我们将序列拆分成单个 FASTA 放到一个目录里
    single_dir = FASTA_OUTPUT.parent / "single_fastas"
    single_dir.mkdir(exist_ok=True)
    for uid, record in sequences.items():
        seq = record["sequence"] if isinstance(record, dict) else record
        (single_dir / f"{uid}.fasta").write_text(f">{uid}\n{seq}\n")

    print(f"单条 FASTA 已生成: {single_dir}")
    print("TMHMM2 在线版: http://www.cbs.dtu.dk/services/TMHMM-2.0/")
    print("(需要逐条提交，或使用 Phobius 批量版替代)\n")


if __name__ == "__main__":
    main()
