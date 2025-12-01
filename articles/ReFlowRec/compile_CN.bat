@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo ========================================
echo Compiling Chinese LaTeX document with XeLaTeX
echo ========================================
echo.
echo Step 1: First compilation with xelatex...
xelatex.exe -synctex=1 -interaction=nonstopmode ReFlowRec_CN.tex
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo First compilation failed! Check the error messages above.
    pause
    exit /b 1
)

echo.
echo Step 2: Running bibtex...
bibtex.exe ReFlowRec_CN
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo BibTeX completed with warnings (this is usually OK).
)

echo.
echo Step 3: Second compilation with xelatex...
xelatex.exe -synctex=1 -interaction=nonstopmode ReFlowRec_CN.tex

echo.
echo Step 4: Third compilation with xelatex (for cross-references)...
xelatex.exe -synctex=1 -interaction=nonstopmode ReFlowRec_CN.tex

echo.
echo ========================================
echo Compilation complete!
echo Output file: ReFlowRec_CN.pdf
echo ========================================
pause

