#!/usr/bin/env python3
"""
Compile LaTeX files for submission package
Requires: pdflatex, bibtex (MiKTeX or TeX Live)
"""
import subprocess
import sys
from pathlib import Path

def compile_latex(source_file, output_dir=None):
    """Compile a LaTeX file to PDF"""
    source_path = Path(source_file)
    if not source_path.exists():
        print(f"[ERROR] File not found: {source_file}")
        return False

    working_dir = source_path.parent
    filename = source_path.stem

    print(f"\n[Compiling] {source_file}")
    print("=" * 60)

    # Compile twice for references
    for i in range(2):
        print(f"\nPass {i+1}/2...")
        result = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', filename],
            cwd=working_dir,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"[WARNING] pdflatex returned non-zero exit code")
            # Print last 20 lines of error
            lines = result.stdout.split('\n')
            print("\n".join(lines[-20:]))

    # Check if PDF was created
    pdf_file = working_dir / f"{filename}.pdf"
    if pdf_file.exists():
        print(f"\n[OK] PDF created: {pdf_file}")
        return True
    else:
        print(f"\n[ERROR] PDF creation failed")
        return False

def main():
    print("=" * 60)
    print("LaTeX Compilation Script")
    print("=" * 60)

    files_to_compile = [
        "main_text/main_manuscript.tex",
        "supplementary/supplementary_materials.tex",
        "cover_letter/cover_letter.tex"
    ]

    results = []
    for tex_file in files_to_compile:
        success = compile_latex(tex_file)
        results.append((tex_file, success))

    print("\n" + "=" * 60)
    print("Compilation Summary")
    print("=" * 60)
    for tex_file, success in results:
        status = "✓ OK" if success else "✗ FAILED"
        print(f"  {status}: {tex_file}")

if __name__ == "__main__":
    main()
