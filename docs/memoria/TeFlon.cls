\NeedsTeXFormat{LaTeX2e}
\ProvidesPackage{TeFlon}[2019/03/24]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% AQUI SE TIENE LO NECESARIO Y MÁS
% 
% Ambientes de cuadros
% Dibujar Funciones
% Dibujar Tableros
% Creación de código fuente
% estilos, tamaños, cabeceras y pie de página
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Aqui se incluyen un conjunto de instrucciones y paquetes
% necesarios para todo lo que se quiera escribir
% NOTA: es evidente que el creador de la plantilla NO HA ESCRITO
% NINGUN PAQUETE DE LOS DE ABAJO
%
% Ha recopilado y modificado todo lo que él considera necesario
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\LoadClass[12pt, a4paper, oneside, ,table]{book}
\usepackage{ifthen}

\newcommand*{\imprimir}[1]{
\ifthenelse{\equal{#1}{si}}{\newgeometry{margin=1in, twoside}}{}
  }
\renewcommand\thepart{\Alph{part}}
\usepackage[spanish]{babel}
\usepackage[utf8]{inputenc}
\usepackage[margin=1in]{geometry}
\usepackage{lipsum}
\usepackage[acronym,shortcuts]{glossaries}
\makeglossaries
\input{Data/acronimos.tex}
\usepackage{xparse}
\usepackage{multirow}
\usepackage{amsmath,amsthm,amssymb}
\usepackage{booktabs}

\usepackage{tikz}
\usepackage{emptypage}
\usepackage{esvect}
\usetikzlibrary{arrows,positioning,fit,shapes,calc}
\usepackage{framed}
\usepackage{fancyhdr} 
\usepackage[
    type={CC},
    modifier={zero},
    version={1.0},
]{doclicense}
\pagestyle{fancy} 
\usepackage{dtk-logos}
\usepackage{verbatim}
\usepackage{eurosym}
\usepackage{multicol}
\usepackage{multirow}
\usepackage{amsmath,amsthm,amssymb}
\usepackage{mdframed}
\usepackage[normalem]{ulem}
\useunder{\uline}{\ul}{}
\usepackage{tikz}
\usepackage{pst-plot}
\usepackage{graphicx}
\usepackage{pgfplots}
\usepackage{pdfpages}
\usepackage{fancyhdr}
\usepackage[table,xcdraw]{xcolor}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%CAMBIAR X1 Z1 Y A1 para el documento de TFG
\lhead[\grado]{\ntrabajo}
\rhead[\facultad]{UCM}
\renewcommand{\headrulewidth}{0.5pt}
\lfoot[\npersona]{}
\renewcommand{\footrulewidth}{0.5pt}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\usetikzlibrary{calc,angles,positioning,intersections,quotes,decorations.markings}
\usepackage{pgfplots}
\usepackage{graphicx}
\usepackage{tikz}
\usetikzlibrary{shapes,arrows}
\usepackage{subfigure}
\usepackage{float}
\usepackage{multicol}
\pgfplotsset{compat=1.5}

%\datos{si}{UCM}{FDI}{grado}{nombre tutor}{nombre propio}{año de finalización}{ciudad donde se realiza}
\newcommand*{\incorporarDatos}[9]{
\ifthenelse{\equal{#1}{si}}{\imprimir{si}}{\imprimir{no}}
\ifthenelse{\equal{#2}{UCM}}{
\newcommand*{\print}{Images/logo_UCM.jpg}
\newcommand*{\facultad}{Facultad de #3}
\newcommand*{\grado}{Grado en #4}
\newcommand*{\tutor}{#5}
\newcommand*{\curso}{#6}
\newcommand*{\ciudad}{#7}
\newcommand*{\npersona}{#8}
\newcommand*{\ntrabajo}{#9}
}{}

}

\tikzstyle{decision} = [diamond, draw, fill=blue!20, 
    text width=4.5em, text badly centered, node distance=3cm, inner sep=0pt]
\tikzstyle{block} = [rectangle, draw, fill=blue!20, 
    text width=5em, text centered, rounded corners, minimum height=4em]
\tikzstyle{line} = [draw, -latex']
\tikzstyle{cloud} = [draw, ellipse,fill=red!20, node distance=3cm,
    minimum height=2em]
%%%%%%%%%%%%%%%%%%%%% Dibujar funciones 
% https://es.sharelatex.com/learn/Pgfplots_package
% http://pgfplots.sourceforge.net/gallery.html
%%%%%%%%%%%%%%%%%%%%% Oficial documentation
% http://pgfplots.sourceforge.net/pgfplots.pdf
% http://pgfplots.sourceforge.net/pgfplotstable.pdf
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\N}{\mathbb{N}}
\newcommand{\Z}{\mathbb{Z}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\Q}{\mathbb{Q}}
\newcommand{\fd}{\rightarrow}
\newcommand{\cont}{\subset}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newenvironment{theorem}[2][Teorema]{\begin{framed}\begin{trivlist}
\item[\hskip \labelsep {\bfseries #1}\hskip \labelsep {\bfseries #2.}]}{\end{trivlist}\end{framed}}
\newenvironment{regla}[2][Regla]{\begin{trivlist}
\item[\hskip \labelsep {\bfseries #1}\hskip \labelsep {\bfseries #2.}]}{\end{trivlist}}
\newenvironment{axioma}[2][Axioma]{\begin{trivlist}
\item[\hskip \labelsep {\bfseries #1}\hskip \labelsep {\bfseries #2.}]}{\end{trivlist}}
\newenvironment{lemma}[2][Lemma]{\begin{trivlist}
\item[\hskip \labelsep {\bfseries #1}\hskip \labelsep {\bfseries #2.}]}{\end{trivlist}}
\newenvironment{exercise}[2][Ejercicio]{\begin{trivlist}
\item[\hskip \labelsep {\bfseries #1}\hskip \labelsep {\bfseries #2.}]}{\end{trivlist}}
\newenvironment{reflection}[2][Reflection]{\begin{trivlist}
\item[\hskip \labelsep {\bfseries #1}\hskip \labelsep {\bfseries #2.}]}{\end{trivlist}}
\newenvironment{proposition}[2][Proposicion]{\begin{trivlist}
\item[\hskip \labelsep {\bfseries #1}\hskip \labelsep {\bfseries #2.}]}{\end{trivlist}}
\newenvironment{corollary}[2][Corolario]{\begin{trivlist}
\item[\hskip \labelsep {\bfseries #1}\hskip \labelsep {\bfseries #2.}]}{\end{trivlist}}

\pgfplotsset{soldot/.style={color=blue,only marks,mark=*}} \pgfplotsset{holdot/.style={color=blue,fill=white,only marks,mark=*}}
\usepackage{mdframed}
\newdimen\arrowsize
\pgfarrowsdeclare{squarea}{squarea}
{
  \arrowsize=0.4pt
  \advance\arrowsize by.275\pgflinewidth%
  \pgfarrowsleftextend{+-\arrowsize}
  \advance\arrowsize by.5\pgflinewidth
  \pgfarrowsrightextend{+\arrowsize}
}
{
  \arrowsize=0.4pt
  \advance\arrowsize by.275\pgflinewidth%
  \pgfsetdash{}{+0pt}
  \pgfsetroundjoin
  \pgfpathmoveto{\pgfqpoint{1\arrowsize}{4\arrowsize}}
  \pgfpathlineto{\pgfqpoint{-7\arrowsize}{4\arrowsize}}
  \pgfpathlineto{\pgfqpoint{-7\arrowsize}{-4\arrowsize}}
  \pgfpathlineto{\pgfqpoint{1\arrowsize}{-4\arrowsize}}
  \pgfpathclose
  \pgfusepathqfillstroke
}
\newcommand{\fantasma}{
\newpage
\thispagestyle{empty}
\phantom{a}
\newpage}
\pgfarrowsdeclare{open squarea}{open squarea}%{{-.5bp}{8.5bp}}
{
  \arrowsize=0.4pt
  \advance\arrowsize by.275\pgflinewidth%
  \pgfarrowsleftextend{+-.5\pgflinewidth}
  \advance\arrowsize by7\arrowsize
  \advance\arrowsize by.5\pgflinewidth
  \pgfarrowsrightextend{+\arrowsize}
}
{
  \arrowsize=0.4pt
  \advance\arrowsize by.275\pgflinewidth%
  \pgfsetdash{}{+0pt}
  \pgfsetroundjoin
  \pgfpathmoveto{\pgfqpoint{8\arrowsize}{4\arrowsize}}
  \pgfpathlineto{\pgfqpoint{0\arrowsize}{4\arrowsize}}
  \pgfpathlineto{\pgfqpoint{0\arrowsize}{-4\arrowsize}}
  \pgfpathlineto{\pgfqpoint{8\arrowsize}{-4\arrowsize}}
  \pgfpathclose
  \pgfusepathqstroke
}


\makeatletter
\newdimen\tempa
\newdimen\tempb
\pgfdeclareshape{diamond in circle}{
\inheritsavedanchors[from=diamond] 
\inheritsavedanchors[from=circle] 
\inheritanchorborder[from=circle]
\inheritanchor[from=circle]{center}
\inheritanchor[from=circle]{radius}
\inheritanchor[from=circle]{north}
\inheritanchor[from=circle]{south}
\inheritanchor[from=circle]{east}
\inheritanchor[from=circle]{west}
\inheritanchor[from=circle]{anchorborder}
  \saveddimen\radius{

    \pgf@ya=.5\ht\pgfnodeparttextbox%
    \advance\pgf@ya by.5\dp\pgfnodeparttextbox%
    \pgfmathsetlength\pgf@yb{\pgfkeysvalueof{/pgf/inner ysep}}%
    \advance\pgf@ya by\pgf@yb%

    \pgf@xa=.5\wd\pgfnodeparttextbox%
    \pgfmathsetlength\pgf@xb{\pgfkeysvalueof{/pgf/inner xsep}}%
    \advance\pgf@xa by\pgf@xb%

    \pgf@process{\pgfpointnormalised{\pgfqpoint{\pgf@xa}{\pgf@ya}}}%
    \ifdim\pgf@x>\pgf@y%
        \c@pgf@counta=\pgf@x%
        \ifnum\c@pgf@counta=0\relax%
        \else%
          \divide\c@pgf@counta by 255\relax%
          \pgf@xa=16\pgf@xa\relax%
          \divide\pgf@xa by\c@pgf@counta%
          \pgf@xa=16\pgf@xa\relax%
        \fi%
      \else%
        \c@pgf@counta=\pgf@y%
        \ifnum\c@pgf@counta=0\relax%
        \else%
          \divide\c@pgf@counta by 255\relax%
          \pgf@ya=16\pgf@ya\relax%
          \divide\pgf@ya by\c@pgf@counta%
          \pgf@xa=16\pgf@ya\relax%
        \fi%
    \fi%
    \pgf@x=\pgf@xa%

    \pgfmathsetlength{\pgf@xb}{\pgfkeysvalueof{/pgf/minimum width}}%
    \pgfmathsetlength{\pgf@yb}{\pgfkeysvalueof{/pgf/minimum height}}%
    \ifdim\pgf@x<.5\pgf@xb%
        \pgf@x=.5\pgf@xb%
    \fi%
    \ifdim\pgf@x<.5\pgf@yb%
        \pgf@x=.5\pgf@yb%
    \fi%

    \pgfmathsetlength{\pgf@xb}{\pgfkeysvalueof{/pgf/outer xsep}}%
    \pgfmathsetlength{\pgf@yb}{\pgfkeysvalueof{/pgf/outer ysep}}%
    \ifdim\pgf@xb<\pgf@yb%
      \advance\pgf@x by\pgf@yb%
    \else%
      \advance\pgf@x by\pgf@xb%
    \fi%
  }
\backgroundpath{
    \tempa=\radius
    \pgfmathsetlength{\pgf@xb}{\pgfkeysvalueof{/pgf/outer xsep}}%
    \pgfmathsetlength{\pgf@yb}{\pgfkeysvalueof{/pgf/outer ysep}}%
    \ifdim\pgf@xb<\pgf@yb%
      \advance\tempa by-\pgf@yb%
    \else%
      \advance\tempa by-\pgf@xb%
    \fi%
    \pgfpathmoveto{\centerpoint\advance\pgf@x by\radius}%
    \pgfpathlineto{\centerpoint\advance\pgf@y by\radius}%
    \pgfpathlineto{\centerpoint\advance\pgf@x by-\radius}%
    \pgfpathlineto{\centerpoint\advance\pgf@y by-\radius}%
    \pgfpathclose%
  }
\behindbackgroundpath{
    \tempa=\radius%
    \pgfmathsetlength{\pgf@xb}{\pgfkeysvalueof{/pgf/outer xsep}}%
    \pgfmathsetlength{\pgf@yb}{\pgfkeysvalueof{/pgf/outer ysep}}%
    \ifdim\pgf@xb<\pgf@yb%
      \advance\tempa by-\pgf@yb%
    \else%
      \advance\tempa by-\pgf@xb%
    \fi%
    \pgfpathcircle{\centerpoint}{\tempa}%
  }
}
\makeatother

\newcommand{\parameter}[1]{$\langle\mbox{#1}\rangle$}
\usepackage{enumitem}
\setlength{\parindent}{12pt}
\renewcommand{\theenumi}{\alph{enumi})}

\newcommand{\teflon}{\LARGE${\displaystyle \mathbf {T\!^{{}_{\scriptstyle E}}\!\!\!\!\;\;F\!_{\displaystyle L}\!O\!^{{}_{\scriptstyle N}}}}$}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Más creaciones de Pascal
\usepackage[most]{tcolorbox}
\newtcolorbox{doublebox}[1][]{%
  enhanced jigsaw,
  sharp corners,
  colback=white,
  borderline={1pt}{-2pt}{black},
  fontupper={\setlength{\parindent}{20pt}},
  #1
}
\definecolor{rosso}{RGB}{220,57,18}
\definecolor{giallo}{RGB}{255,153,0}
\definecolor{blu}{RGB}{102,140,217}
\definecolor{verde}{RGB}{16,150,24}
\definecolor{viola}{RGB}{153,0,153}

\makeatletter

\tikzstyle{chart}=[
    legend label/.style={font={\scriptsize},anchor=west,align=left},
    legend box/.style={rectangle, draw, minimum size=5pt},
    axis/.style={black,semithick,->},
    axis label/.style={anchor=east,font={\tiny}},
]

\tikzstyle{bar chart}=[
    chart,
    bar width/.code={
        \pgfmathparse{##1/2}
        \global\let\bar@w\pgfmathresult
    },
    bar/.style={very thick, draw=white},
    bar label/.style={font={\bf\small},anchor=north},
    bar value/.style={font={\footnotesize}},
    bar width=.75,
]

\tikzstyle{pie chart}=[
    chart,
    slice/.style={line cap=round, line join=round, very thick,draw=white},
    pie title/.style={font={\bf}},
    slice type/.style 2 args={
        ##1/.style={fill=##2},
        values of ##1/.style={}
    }
]

\pgfdeclarelayer{background}
\pgfdeclarelayer{foreground}
\pgfsetlayers{background,main,foreground}


\newcommand{\pie}[3][]{
    \begin{scope}[#1]
    \pgfmathsetmacro{\curA}{90}
    \pgfmathsetmacro{\r}{1}
    \def\c{(0,0)}
    \node[pie title] at (90:1.3) {#2};
    \foreach \v/\s in{#3}{
        \pgfmathsetmacro{\deltaA}{\v/100*360}
        \pgfmathsetmacro{\nextA}{\curA + \deltaA}
        \pgfmathsetmacro{\midA}{(\curA+\nextA)/2}

        \path[slice,\s] \c
            -- +(\curA:\r)
            arc (\curA:\nextA:\r)
            -- cycle;
        \pgfmathsetmacro{\d}{max((\deltaA * -(.5/50) + 1) , .5)}

        \begin{pgfonlayer}{foreground}
        \path \c -- node[pos=\d,pie values,values of \s]{$\v\%$} +(\midA:\r);
        \end{pgfonlayer}

        \global\let\curA\nextA
    }
    \end{scope}
}

\newcommand{\legend}[2][]{
    \begin{scope}[#1]
    \path
        \foreach \n/\s in {#2}
            {
                  ++(0,-10pt) node[\s,legend box] {} +(5pt,0) node[legend label] {\n}
            }
    ;
    \end{scope}
}
\definecolor{findOptimalPartition}{HTML}{D7191C}
\definecolor{storeClusterComponent}{HTML}{FDAE61}
\definecolor{dbscan}{HTML}{ABDDA4}
\definecolor{constructCluster}{HTML}{2B83BA}

\usepackage{graphicx}
%\annotatedFigureBoxCustom{bottom-left}{top-right}{label}{label-position}{box-color}{label-color}{border-color}{text-color}
\newcommand*\annotatedFigureBoxCustom[8]{\draw[#5,thick,rounded corners] (#1) rectangle (#2);\node at (#4) [fill=#6,thick,shape=circle,draw=#7,inner sep=2pt,font=\sffamily,text=#8] {\textbf{#3}};}
%\annotatedFigureBox{bottom-left}{top-right}{label}{label-position}
\newcommand*\annotatedFigureBox[4]{\annotatedFigureBoxCustom{#1}{#2}{#3}{#4}{white}{white}{black}{black}}
\newcommand*\annotatedFigureText[4]{\node[draw=none, anchor=south west, text=#2, inner sep=0, text width=#3\linewidth,font=\sffamily] at (#1){#4};}
\newenvironment {annotatedFigure}[1]{\centering\begin{tikzpicture}
\node[anchor=south west,inner sep=0] (image) at (0,0) { #1};\begin{scope}[x={(image.south east)},y={(image.north west)}]}{\end{scope}\end{tikzpicture}}

\newtcolorbox{cosobox}[2][]{colbacktitle=red!10!white,
colback=blue!10!white,coltitle=red!70!black,
title={#2},fonttitle=\bfseries,#1}

\newcommand*\circled[1]{
    \tikz[baseline=(char.base)]{
        \node[shape=circle,draw,inner sep=0pt] (char) {#1\strut}
    }\kern-3pt
}
\usepackage{bbding}
\usepackage{pifont}
\usepackage{wasysym}
\usepackage{amssymb}

\newcommand{\xmark}{\ding{55}}%

\psset{showbbox=false,treemode=R,linewidth=0.3pt,treesep=4ex,levelsep=2cm}
\newcommand{\LFTw}[2]{%
\Tr[ref=#1]{\psframebox[linestyle=none,framesep=4pt]{%
\begin{varwidth}{15ex}\center #2\end{varwidth}}}}
\newcommand{\LFTr}[2]{\Tr[ref=#1]{\psframebox[linestyle=none,framesep=4pt]{#2}}}

\def\pstreehooki{\psset{thislevelsep=*0pt}}
\def\pstreehookiii{\psset{thislevelsep=*0pt}}
\def\pstreehookv{\psset{thislevelsep=*0pt}}
\def\pstreehookvii{\psset{thislevelsep=*0pt}}
\def\pstreehookix{\psset{thislevelsep=*0pt}}

\newcommand{\pascal}{\LARGE${\displaystyle \mathbf {P\!\!^{{}_{\scriptstyle A}}\!\!\!\!\!\;\;S\!_{\displaystyle C}\!A\!\!^{{}_{\scriptstyle L}}}}$}

\usepackage[most]{tcolorbox}
\usepackage{listings}
\definecolor{keywords}{HTML}{8A4A0B}
\definecolor{background}{HTML}{EEEEEE}
\definecolor{comments}{RGB}{128,0,0}
\definecolor{myblue}{RGB}{0,51,204}
\definecolor{halfgray}{gray}{0.55}
\definecolor{rojo}{RGB}{230,0,0}
\usepackage[scaled=.95]{inconsolata}
\tcbuselibrary{listings, breakable, skins}
\definecolor{bg}{rgb}{0.85,0.85,0.85}
%\newtcblisting{ccode}[2][]

\lstdefinelanguage{PLSQL}[]{SQL}{
    morekeywords={ENUM,IF,REFERENCES,DECLARE,BEGIN,
    ELSIF,WHILE,LOOP,EXIT,FOR,REVERSE,OPEN,FETCH,CLOSE,
    IS,CURSOR,RAISE,REPLACE,PROCEDURE,CALL,RETURN,CONSTANT,
    REFERENCING,OLD,NEW,AFTER,BEFORE,EACH,ROW,MINUS},
    deletekeywords={ACTION} }[keywords,comments,strings]
\lstset{language=PLSQL,
    keywordstyle=\color{rojo},
    texcsstyle=*\color{myblue},
    basicstyle=\textbf\normalfont\ttfamily,
    commentstyle=\color{comments}\ttfamily,
    stringstyle=\rmfamily,
    numbers=left,
    numberstyle=\scriptsize,
    stepnumber=1,
    numbersep=8pt,
    captionpos=top,
    showstringspaces=false,
    breaklines=true,
    frameround=ftff,
    morekeywords={RequirePackage,ProvidesPackage,NeedsTeXFormat},
    backgroundcolor=\color{background},
    literate=
            *{\{}{{\textcolor{myblue}{\{}}}{1}
            {\}}{{\textcolor{myblue}{\}}}}{1}
            {\\}{{\textcolor{myblue}{\textbackslash}}}{1}
            {\$}{{\textcolor{rojo}{\$}}}{1}
            {\&}{{\textcolor{rojo}{\&}}}{1}
            {documentclass}{{\textcolor{rojo}{\textbackslash documentclass}}}{12}
            {\%\\documentclass}{{\textcolor{rojo}{\textbackslash documentclass}}}{12}
            {\%\\usepackage}{{\textcolor{rojo}{\textbackslash usepackage}}}{9}
            {[}{{\textcolor{myblue}{[}}}{1}
            {]}{{\textcolor{myblue}{]}}}{1}
            {á}{{\'a}}1 {é}{{\'e}}1 {í}{{\'i}}1 {ó}{{\'o}}1 {ú}{{\'u}}1 {ñ}{{\~n}}1 {Á}{{\'A}}1 {É}{{\'E}}1 {Í}{{\'I}}1 {Ó}{{\'O}}1 {Ú}{{\'U}}1,
    frame=single,
    frameround={t}{t}{t}{t},
    framexleftmargin=6mm,
    numbers=left,
    numberstyle=\tiny\color{halfgray},
}


\newcommand{\wlatex}{\lstinline}

\newcommand{\parameter}[1]{$\langle\mbox{#1}\rangle$}

\newenvironment{truco}[2][Truco de resolución o notas]{\begin{mdframed}\begin{trivlist}
\item[\hskip \labelsep {\bfseries #1}\hskip \labelsep {\bfseries #2.}]}{\end{trivlist}\end{mdframed}}

\usepackage[parfill]{parskip}
\newcommand*{\tituloTFG}[0]{
\pagenumbering{roman}
\thispagestyle{empty}
\vspace*{5mm}
\begin{center}
    \rule[0.5ex]{\linewidth}{2pt}\vspace*{-\baselineskip}\vspace*{3.2pt}\\
    \rule[0.5ex]{\linewidth}{2pt}\\
    [2mm]
    {\textbf{\LARGE{TEFLON X}} }\\[3mm] %Titulo del TFG
    
    \rule[0.5ex]{\linewidth}{1pt}\vspace*{-\baselineskip}\vspace{3.2pt}
    \rule[0.5ex]{\linewidth}{2pt}\\
    \vspace{6.5mm}
    {\large Por\\\npersona}
    \vspace{6.5mm}
    {\large\textsc{}}\\ %Nombres de los autores
    \vspace{5mm}
    \includegraphics[scale=0.3]{\print}\\ %Logo de la UCM
    \vspace{6mm}
    {\large \grado\\    %Departamento del profesor que es el tutor
    \textsc{\facultad}}\\ %Facultad para la que se hace
    \vspace{11mm}
    \begin{minipage}{10cm}
    \begin{center}
    \tutor\\
      \textbf{título en inglés}\\
      \vspace{2mm}
    \end{center}
    \vspace{4mm}
    
    %Cuadro de presentación del TFG
    \end{minipage}\\
    \vspace{4mm}
    {\large\textsc{\ciudad, \curso}} %Fecha de finalización o presentación
    \vspace{12mm}
\end{center}
}
\newcommand*{\abstractT}[0]{
\newpage
\chapter*{Sobre \teflon X}
\noindent
\textsc{Teflon X(cc0 1.0(documentación) MIT(código))%\cite{dPaciosTeflon}% 
es una plantilla de \LaTeX\phantom{} creada por David Pacios Izquierdo con fecha de Enero de 2018. Con atribuciones de uso CC0.}\\\\
\noindent
Esta plantilla fue desarrollada para facilitar la creación de documentación profesional para Trabajos de Fin de Grado, Trabajos de Fin de Máster o Doctorados. La versión usada es la X\\\\
\noindent
V:\textsc{X Overleaf V2 with XeLaTeX, margin 1in, bib}
\vfill
\noindent
\textsc{\textbf{\underline{Contacto}}\\ \textbf{Autor:} David Pacios Izquiero \\ \textbf{Correo:} \url{dpacios@ucm.es}\\ \textbf{ASCII:} \url{ascii@ucm.es}\\ Despacho 110 - Facultad de Informática}
}

\newcommand*{\indiceTFG}[0]{
\abstractT
    \addtocontents{toc}{\hfill \textbf{Página} \par}
\tableofcontents
}
\newcommand*{\biblioTFG}[1]{
\newpage
\addcontentsline{toc}{chapter}{\protect\numberline{#1}Bibliografía y enlaces de referencia}%0
\nocite{*}
\bibliographystyle{ieeetr}
\bibliography{biblio.bib}
}
\newcommand*{\indiceFiguras}[0]{\listoffigures}
\newcommand*{\indiceTablas}[0]{\listoftables}
\newcommand*{\licenciaTFG}[3]{
\newpage
\thispagestyle{empty}
\vspace*{\fill}
\texttt{#1\\}
\texttt{#2\\}
\texttt{Ult. actualización \today}\\
\texttt{\LaTeX{} {lic. LPPL} \& powered by {\small{\teflon{}}} \textsc{{#3}}}
\doclicenseThis
}

\newcommand*{\hill}[2]{
\begin{list}{}{
     \setlength{\leftmargin}{0.5\textwidth}
     \setlength{\parsep}{0cm}
     \addtolength{\topsep}{0.5cm}}
   \item \begin{flushright}\small\em #1
         \end{flushright}
   \item \begin{flushright}\small#2
         \end{flushright}
\end{list}
}
\newcommand{\capitulo}[2]{
    \chapter*{Capítulo #1\vspace{1cm}\\#2}    \addcontentsline{toc}{chapter}{\protect\numberline{#1}#2}}
}
\newcommand{\parte}[2]{
\part*{Parte #1\vspace{1cm}\\#2} \addcontentsline{toc}{part}{\protect\numberline{#1}\hspace{-3mm}#2} 
}
\newcommand*{\clp}[]{
    \textcolor{white}{t}
}
\usepackage{xcolor}
\definecolor{good_black}{RGB}{36, 30, 32}
\definecolor{good_green}{RGB}{109, 191, 81}
%\pagecolor{good_black}
%\color{good_green}

%Flowchart
\tikzstyle{startstop} = [rectangle, rounded corners, minimum width=3cm, minimum height=1cm,text centered, draw=black, fill=red!30]
\tikzstyle{io} = [trapezium, trapezium left angle=70, trapezium right angle=110, minimum width=3cm, minimum height=1cm, text centered, draw=black, fill=blue!30]
\tikzstyle{process} = [rectangle, minimum width=3cm, minimum height=1cm, text centered, draw=black, fill=orange!30]
\tikzstyle{decision} = [diamond, minimum width=2cm, minimum height=1cm, text centered, draw=black, fill=green!30]
\tikzstyle{process} = [rectangle, minimum width=3cm, minimum height=1cm, text centered, draw=black, fill=orange!30]
\tikzstyle{arrow} = [thick,->,>=stealth]

\endinput