@echo off
chcp 65001 >nul
echo ============================================================
echo Compiling LaTeX files for Submission Package
echo ============================================================
echo.

cd /d "%~dp0"

echo [1/3] Compiling main manuscript...
cd main_text
pdflatex -interaction=nonstopmode main_manuscript.tex >nul 2>&1
pdflatex -interaction=nonstopmode main_manuscript.tex >nul 2>&1
if exist "main_manuscript.pdf" (
    echo      ✓ main_manuscript.pdf created
) else (
    echo      ✗ Compilation failed
)
cd ..

echo.
echo [2/3] Compiling supplementary materials...
cd supplementary
pdflatex -interaction=nonstopmode supplementary_materials.tex >nul 2>&1
pdflatex -interaction=nonstopmode supplementary_materials.tex >nul 2>&1
if exist "supplementary_materials.pdf" (
    echo      ✓ supplementary_materials.pdf created
) else (
    echo      ✗ Compilation failed
)
cd ..

echo.
echo [3/3] Compiling cover letter...
cd cover_letter
pdflatex -interaction=nonstopmode cover_letter.tex >nul 2>&1
pdflatex -interaction=nonstopmode cover_letter.tex >nul 2>&1
if exist "cover_letter.pdf" (
    echo      ✓ cover_letter.pdf created
) else (
    echo      ✗ Compilation failed
)
cd ..

echo.
echo ============================================================
echo Compilation complete!
echo ============================================================
echo.
echo Generated PDFs:
echo   - main_text/main_manuscript.pdf
echo   - supplementary/supplementary_materials.pdf
echo   - cover_letter/cover_letter.pdf
echo.
echo Figures (ready to use):
echo   - figures/*.pdf
echo.
pause
