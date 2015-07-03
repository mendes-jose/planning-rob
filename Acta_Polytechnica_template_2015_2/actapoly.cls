%\listfiles
%% Class file actapoly.cls
%% provide style for journal Acta Polytechnica
%% http://ctn.cvut.cz/ap/
%%
%% Acta Polytechnica is published by
%% Czech Technical University in Prague
%% Czech Technical University in Prague - Central Library
%% Technicka 6, 160 80 Prague 6, Czech Republic
%% http://www.cvut.cz/
%%
%% This LaTeX class file is based on class article.cls
%%
%% This class uses the following packages available on http://ctan.org/ :
%% amsfonts, amsmath, amssymb, amsthm, babel, booktabs, caption, enumitem, environ, fancyhdr
%% float, fontenc, geometry, graphicx, hyperref, kvoptions, lastpage, lmodern, mathtools, microtype,
%% natbib, textcomp, titlesec, url
%%
%% See the line \ProvidesClass for version and release date information
%%
%% Author: Tomas Hejda (Tom\'a\v s Hejda) (tohecz@gmail.com)
%%
%% Copyright holder: 
%% Czech Technical University in Prague
%%
\newif\ifX@c % standing for if-class
\@ifclassloaded{actapolyissue}{\X@cfalse}{\X@ctrue}

\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{actapoly}[2015/02/02 v0.5 LaTeX class for Acta Polytechnica journal]

% error, warning, note and info handling
\def\X@EWNI@add#1#2{{%
  \def\MessageBreak{ }%
  \def\space{ }%
  \edef\X@tempa{\noexpand\g@addto@macro{\noexpand\X@EWNI@container}{\noexpand\item\noexpand\textbf{#1:} #2}}%
  \X@tempa
}}
\let\X@EWNI@container\@empty
\AtEndDocument{\ifX@nosilentend
  \ifx\X@EWNI@container\@empty\else
    \ttl@straightclass{subsection}*{\color{red}\LaTeX\ class errors and warnings}%
    \begin{itemize}\color{red}%
      \X@EWNI@container
    \end{itemize}%
  \fi
\fi}
\def\X@Error#1{%
  \X@EWNI@add{Error}{#1}%
  \GenericError{(actapoly)\@spaces\@spaces\@spaces}{actapoly.cls Error: #1}%
  }
\def\X@Warning#1{%
  \X@EWNI@add{Warning}{#1}%
  \GenericWarning{(actapoly)\@spaces\@spaces\@spaces}{actapoly.cls Warning: #1}%
  }
\def\X@Note#1{%
  \GenericWarning{(actapoly)\@spaces\@spaces}{actapoly.cls Note: #1}%
  }
\def\X@Info#1{%
  \GenericInfo{(actapoly)\@spaces\@spaces}{actapoly.cls Info: #1}%
  }

\PassOptionsToPackage{patch}{kvoptions}
\RequirePackage{kvoptions}

\newif\ifX@review\X@reviewtrue
\newif\ifX@showdate\X@showdatetrue
\newif\ifX@injournal\X@injournalfalse
\newif\ifX@eprint\X@eprintfalse
\newif\ifX@onecolumn\X@onecolumnfalse
\newif\ifX@complain\X@complaintrue
\newif\ifX@nosilentend\X@nosilentendtrue
\newif\ifX@onecolumn\X@onecolumnfalse
\newif\ifX@natbib\X@natbibtrue

\DeclareBoolOption{eprint}
\DeclareBoolOption{@proofreading}
\DeclareBoolOption{@bwdir}
\DeclareBoolOption[true]{review}
\DeclareBoolOption[true]{natbib}
\DeclareBoolOption{arXiv}
\DeclareBoolOption{approc}

%\ifX@c%@c

\def\X@clsoptions{}
\DeclareVoidOption{draft}{\xdef\X@clsoptions{\X@clsoptions,draft}}
\DeclareVoidOption{xdraft}{\xdef\X@clsoptions{\X@clsoptions,draft}}
\DeclareVoidOption{final}{\xdef\X@clsoptions{\X@clsoptions,final}}
\DeclareVoidOption{showdate}{\X@showdatetrue}
\DeclareVoidOption{@nocomplain}{\X@complainfalse\X@nosilentendfalse}
\DeclareVoidOption{@silentend}{\X@nosilentendfalse}
\DeclareVoidOption{@onecolumn}{\X@onecolumntrue}
\DeclareStringOption{@shortenlast}
  \def\actapoly@@shortenlast{}
\DeclareStringOption{@shortenhead}
  \def\actapoly@@shortenhead{0.0}
\DeclareStringOption{@doi}
  \def\actapoly@@doi{}
\DeclareStringOption{@vnyp}
  \def\actapoly@@vnyp{}
\DeclareVoidOption{_showkeys}{
  \RequirePackage[color]{showkeys}
  \definecolor{refkey}{rgb}{1,0,0}
  \definecolor{labelkey}{rgb}{0,1,0}
}
\DeclareStringOption{_injournal}
  \def\actapoly@@injournal{}
\DeclareStringOption{_articlename}
  \def\actapoly@@articlename{}
  
%\fi

\ProcessKeyvalOptions*

\ifactapoly@approc%@approc
  \def\X@journal{AP Proceedings}
\else%@approc
  \def\X@journal{Acta Polytechnica}
\fi%@approc

\ifactapoly@arXiv%@arXiv
  \actapoly@reviewfalse
  \X@showdatefalse
  \actapoly@eprinttrue
  \actapoly@@proofreadingfalse
  \let\X@EWNI@container\@empty
  \let\X@EWNI@add\@gobbletwo
\fi%@arXiv

%%%%%%%%%% EMBEDDING IN JOURNAL %%%%%%%%%%

% store [_injounral] and [_articlename]
\xdef\X@injournal{\csname actapoly@_injournal\endcsname}
\xdef\X@articlename{\csname actapoly@_articlename\endcsname}
% try to get [_articlename] from the shell-escape if [_injournal] is given without it
\ifx\X@injournal\@empty\else%@injournal
  \X@injournaltrue
  \ifx\X@articlename\@empty%@articlename
    \ifnum\pdfshellescape=1\relax
      % in a group, we change catcodes !,[,] -> 0,1,2 and \,{,} -> 12
      \begingroup
        \catcode`\!0
        \catcode`\[1
        \catcode`\]2
        !catcode`!\12
        !catcode`!{12
        !catcode`!}12
        % pwd combined with sed, to strip `/foo/bar/` -> `\def\X@swd{bar}` and save in a .swd file
        !immediate!write18[pwd | sed -e 's:.*/\(.\)\|/$:\1:g' | sed -e 's:\(.*\):\\def\\X@swd{\1}:' > !jobname.swd]
      !endgroup
      % just a check whether the file really exists, or we throw an error
      \IfFileExists{\jobname.swd}{
        % input the file
        \input{\jobname.swd}
        % use the information in \X@swd to set what `[_articlename]` should say
        \edef\X@articlename{\X@swd}
      }{
        %
        \X@Error{Fetching `_articlename`\MessageBreak class option from -shell-escape went wong}{}{}
      }
    \else%@shellescape
      \X@Error{Fetching `_articlename`:\MessageBreak -shell-escape seems to be disabled}{}{}
    \fi%@shellescape
  \fi%@articlename
\fi%@injournal

\ifX@c%@c

% counter that will store the total number of pages
\newcounter{numpages}
% this all has to be done after .aux is read
\AtBeginDocument{
  % make sure these two cross-referencing commands are defined, at least to \relax
  \@ifundefined{r@FirstPage}{\relax}{\relax}
  \@ifundefined{r@LastPage}{\relax}{\relax}
  % \X@rtonum gets the page number from an \r@... command
  \@ifpackageloaded{hyperref}{\def\X@rtonum#1#2#3#4#5{#2}}{\def\X@rtonum#1#2{#2}}
  % start with zero
  \setcounter{numpages}{0}
  % define \X@FirstPage to store the first page number
  \ifx\r@FirstPage\relax
    % we store 0 if nothing better is availible
    \gdef\X@FirstPage{0}
  \else
    % we store the real value of \pageref{FirstPage}, just the problem is that \pageref itself is unexpandable
    \xdef\X@FirstPage{\expandafter\X@rtonum\r@FirstPage}
    % the larger FirstPage, the smaller number of pages
    \setcounter{numpages}{-\X@FirstPage}
    \stepcounter{numpages}
  \fi
  % the same again for LastPage
  \ifx\r@LastPage\relax
    \gdef\X@LastPage{0}
  \else
    \xdef\X@LastPage{\expandafter\X@rtonum\r@LastPage}
    % the larger LastPage, the larger number of pages
    \addtocounter{numpages}{\X@LastPage}
  \fi
  % \AP@page stores the correct first page number
  \ifdefined\AP@page
    \setcounter{page}{\AP@page}
  \fi
}
% \AP@{<article>}{<key>}{<value>} is stored in the file <issue>.ap and is loaded at the end of the class-file
\def\AP@#1#2#3{
  \begingroup
    % store the <article> key
    \edef\reserved@a{#1}%
    % define \AP@<key> = <value> but only if <article> is correct
    \ifx\reserved@a\X@articlename\relax
      \expandafter\xdef\csname AP@#2\endcsname{#3}%
    \fi
  \endgroup
}
% \AP@all is the same as \AP@{<this-article>}
\def\AP@all{\AP@\X@articlename}
% default values for volume, number, year and page
\def\AP@volume{00}%
\def\AP@number{0}%
\def\AP@year{0000}%
\def\AP@page{1}%
% empty \AP@fromissue, it has to be defined
\def\AP@fromissue{}%
% \volumenumberyearpage<volume>:<number>:<year>:<page>: sets the values manually
\def\volumenumberyearpage#1:#2:#3:#4:{
  \def\AP@volume{#1}
  \def\AP@number{#2}
  \def\AP@year{#3}
  \def\AP@page{#4}
  % as well, we modify the headings: show the weblink and hide the "PREPRINT" text
  \let\X@head@preprint\@gobble
  \let\X@hdonline\X@hdonline@
}
% if [@vnyp] class option (volume, number, year, page) is present, use it
\ifx\actapoly@@vnyp\@empty\else%@vnyp
  \expandafter\volumenumberyearpage\actapoly@@vnyp:
  \actapoly@reviewfalse
\fi%@vnyp

% a lot of code dealing with article already embedde in jounral
\ifX@injournal%@injournal
  % try to load the file <journal>.ap
  \IfFileExists{../\X@injournal.ap}{%
    \X@Note{Yes file `../\X@injournal`}\input{../\X@injournal.ap}%
  }{%
    \X@Note{No file `../\X@injournal`}%
  }%
  % writes the key #1 with the value #2 into the article file <article>.ap
  \def\X@AP@write#1#2{\protected@write\X@writeAP{\relax}{\protect\AP@{\X@articlename}{#1}{#2}}}
  % macro for writing the right information into <article>.ap
  \def\X@writejournal{
    % \X@writeAP is open for writing, the filename is <directoryname>.ap
    \newwrite\X@writeAP
    \immediate\openout\X@writeAP=\X@articlename.ap\relax
    % write: total number of pages
    \X@AP@write{numpages}{\arabic{numpages}}
    % write: first page
    \X@AP@write{page}{\AP@page}
    \ifx\actapoly@@doi\@empty\else
      % write: doi
      \X@AP@write{doi}{\actapoly@@doi}
    \fi
    \X@AP@write{injournal}{\X@injournal}
    % write: eprint flag
    \X@AP@write{eprint}{\ifX@eprint1\else0\fi}
    % write: title, shorttitle, author, shortauthor
    \X@AP@write{title}{\@title}
    \X@AP@write{shorttitle}{\@shorttitle}
    \X@AP@write{author}{\@cleanauthor}
    \X@AP@write{shortauthor}{\@shortauthor}
    \X@AP@write{shortauthors}{\@shortauthors}
    \@ifundefined{X@email}{}{\X@AP@write{email}{\X@email}}
    % write: jobname (necessary for inclusion of the PDF file)
    \X@AP@write{jobname}{\jobname}
    % write: abstract
    \X@AP@write{abstract}{\@abstractbody}
    \begingroup
      % write: pagerange
      \X@AP@write{pagerange}{\X@FirstPage-\X@LastPage}
      \X@AP@write{lastpage}{\X@LastPage}
      \@tempcnta\numexpr\X@LastPage-\X@FirstPage+1\relax
      \X@AP@write{numpages}{\the\@tempcnta}
    \endgroup
  }
  % write the data for the journal once they are available (which is at \maketitle )
  \AtBeginDocument{\g@addto@macro{\maketitle}{\X@writejournal}}
\fi%@injournal

\fi%@c

%%% OPTION EXECUTION %%%
% (started by \ProcessKeyvalOptions* above)

% in \AP@fromissue some instructions of the form \actapoly@<key>(true|false) can be stored
\ifX@c%@c
  \AP@fromissue
  \begingroup
    \expandafter\let\expandafter\X@foo\csname AP@doi\endcsname % to make sure that \AP@doi is defined
    \ifx\relax\AP@doi\else\ifx\actapoly@@doi\@empty%doi
      \global\let\actapoly@@doi\AP@doi
    \fi\fi%doi
  \endgroup
\fi%@c

% in the case the [@proofreading] option is enabled, we print ugly proofreading text into the article
\ifactapoly@@proofreading%@proofreading
  % this has to be done after headings are defined, so better \AtBeginDocument
  \AtBeginDocument{%
    % inner foots in {fancy} style
    % \hbox to 0pt{\smash ... \hss}~ to make it occupy no space therefore not modify anything in the document
    \fancyfoot[LO]{\hbox to 0pt{\smash{\Large\sffamily\bfseries Proofreading version!!!}\hss}~}%
    \fancyfoot[RE]{~\hbox to 0pt{\hss\smash{\Large\sffamily\bfseries Proofreading version!!!}}}%
    % inner foots in {plain} style (used for the first page)
    \g@addto@macro\ps@plain{
      \fancyfoot[LO]{\hbox to 0pt{\smash{\Large\sffamily\bfseries Proofreading version!!!}\hss}~}%
      \fancyfoot[RE]{~\hbox to 0pt{\hss\smash{\Large\sffamily\bfseries Proofreading version!!!}}}%
    }
  }
\fi%@proofreading

% [eprint] activates \ifX@eprint, [natbib] activates \ifX@natbib
\let\ifX@eprint\ifactapoly@eprint
\let\ifX@natbib\ifactapoly@natbib
\ifX@onecolumn%@onecolumn
  % for [onecolumn], we add onecolumn as an option of \LoadClass{article}
  \xdef\X@clsoptions{\X@clsoptions,onecolumn}
  % \X@xxxcolumn[#1] is used to print \maketitle, in onecolumn, no special effect is needed
  \long\def\X@xxxcolumn[#1]{#1}
\else%@onecolumn
  % for [twocolumn], we add twocolumn as an option of \LoadClass{article}
  \xdef\X@clsoptions{\X@clsoptions,twocolumn}
  % in [twocolumn], \X@xxxcolumn is simply \twocolumn
  \let\X@xxxcolumn\twocolumn
\fi%@onecolumn

\ifactapoly@review%@review
  \X@reviewtrue
\else%@review
  \X@reviewfalse
\fi%@review

%\fi%@c

%%% CLASS AND PACKAGES LOADING

\ifX@c%@c
% loading {article} class with proper options for an article
\LoadClass[a4paper,10pt,twoside\X@clsoptions]{article}
% setting columnsep
\columnsep0.3in
\else%@c
% loading {article} class with proper options for an issue
\LoadClass[a4paper,10pt,twoside\X@clsoptions]{article}
\fi%@c

% general packages:
\RequirePackage{xcolor}
\RequirePackage{environ}
% mathematics:
\RequirePackage{amsmath,amssymb,amsfonts,amsthm}
\RequirePackage{mathtools}
% page control:
\RequirePackage[a4paper, hmargin={0.9in,0.7in},
  headheight=16pt, footskip=24pt, headsep=12pt,
  includeheadfoot, vmargin={0.55in, 0.55in}]{geometry}
\RequirePackage{fancyhdr}
\RequirePackage{lastpage}
% floats:
\RequirePackage{graphicx}
\RequirePackage{caption}
\RequirePackage{float}
\RequirePackage{booktabs}
% fonts:
\RequirePackage[T1]{fontenc}
\RequirePackage{lmodern}
\RequirePackage[english]{babel}
\RequirePackage{microtype}
\IfFileExists{textcase.sty}{\RequirePackage{textcase}}{}
\IfFileExists{textcomp.sty}{\RequirePackage{textcomp}}{}
% lists:
\RequirePackage{enumitem}
% sections:
\RequirePackage{titlesec}
% bibliography
\ifX@natbib
  % if natbib is wanted, load it with [numbers,sort&compress] as is usual in technical sciences
  \RequirePackage[numbers,sort&compress]{natbib}
  % we as well allow only \citep command
  \let\cite\citep
  \let\citet\cite
\fi
% hyperlinks:
\ifX@eprint
  % in [eprint] version, we set numbered PDF navigation bookmarks
  % and title "Acta Polytechnica", which is better than what authors put into the title
  \RequirePackage[final,bookmarksnumbered=true,pdftitle={\X@journal}]{hyperref}
\else
  % in [noeprint] version, we load the package to avoid errors, but we disable all features
  \RequirePackage[draft]{hyperref}
\fi
\RequirePackage{url}

%%%%%%%%%% MATHEMATICS %%%%%%%%%%

% theorems: slanted font
\def\th@plain{%
  \normalfont
  \slshape
  \thm@notefont{}%
}
% definitons and examples: upright font
\def\th@definition{%
  \normalfont
  \thm@notefont{}%
}
% case
\def\th@case{%
  \normalfont
  \thm@notefont{}%
  \thm@headfont{\itshape\let\upshape\relax}%
}
% black square as QED symbol
\def\qedsymbol{\hbox{\rule[0.3ex]{1.0ex}{1.0ex}}}

% redefinition of \thmhead: we want to remove the extra parentheses around note if note comprises \cite s only
\def\thmhead#1#2#3{\thmname{#1}\thmnumber{\@ifnotempty{#1}{ }\@upn{#2}}\thmnote{ {\the\thm@notefont\X@thmnote{#3}}}}
% we count the \cite s in a semaphore-style manner
\newcount\X@thmnote@cnt
\def\X@thmnote@test#1#2{%
  \def\X@thmnote@test@i{#1#2}%
  \def\X@thmnote@test@ii{,\cite}%
  % if ,<some spaces>\cite follows after a complete \cite, we still want to remove the parentheses
  \ifx\X@thmnote@test@i\X@thmnote@test@ii
    \global\advance\X@thmnote@cnt0\relax
  % otherwise, set the semaphore to a large enough value
  \else
    \global\advance\X@thmnote@cnt2\relax
  \fi
  #1#2%
}
\def\X@thmnote#1{%
  \begingroup
  % reset the semaphore
  \global\X@thmnote@cnt0\relax
  % store original \cite
  \let\X@thmnote@cite\cite
  % let's do the nasty stuff in a box to prevent any troubles
  \setbox0\hbox{%
    % test whether note starts with \cite
    \@ifnextchar\cite{%
      % temporarily, \cite should just test whether it is the last thing in the note
      \renewcommand*\cite[2][]{\@ifnextchar\@gobble{%
        % if it's the last thing, we add 1 to the semaphore
        \global\advance\X@thmnote@cnt1\relax
        % we gobble the \@gobble
        \@gobble
      }{%
        % if it's not the last thing, we test whether ,<some spaces>\cite follows
        \X@thmnote@test
      }}%
    }{%
      % be intact if note doesn't start with \cite
      \relax
    }%
    % process the note with the stuff above
    #1%
    % we add \@gobble as a right separator, and 3x \relax because some bad stuff here excepts parameters
    \@gobble\relax\relax\relax
  }%
  % restore \cite
  \let\cite\X@thmnote@cite
  {%
    % all notes are \normalfont
    \normalfont
    % if semaphore=1, no parentheses, otherwise, yes parentheses
    \ifnum\the\X@thmnote@cnt=1\relax#1\else(#1)\fi
  }%
  \endgroup
}

% we allow broken display equations if really necessary
\allowdisplaybreaks[1]

%%%%%%%%%% FONTS %%%%%%%%%%

% add very small stretch into baselineskip
\g@addto@macro{\selectfont}{\baselineskip=1\baselineskip plus 0.03ex\relax}
% new font sizes
% size of \title
\def\titlelarge{\@setfontsize\titlelarge{12.5}{14.0}}
% size of \section
\def\halflarge{\@setfontsize\halflarge{13}{14.4}}
% size of \subsection
\def\halfsmall{\@setfontsize\halfsmall{11}{12.3}}
% size of {acknowledgements} and {thebibliography}, notice the \baselineskip is a bit larger than in \small
\def\loosesmall{\@setfontsize\loosesmall{9}{11.3}}

% couple symbols
\@ifpackageloaded{textcomp}{}{
  \def\textmu{%
    \global\def\textmu{$\mu$}%
    \textmu
    \X@Warning{Package `textcomp` used to typeset \string\micro\space is\MessageBreak
      not available, switching to "poor man's\MessageBreak
      solution"}%
  }
  \def\textdegree{%
    \global\def\textdegree{$^\circ$}%
    \textdegree
    \X@Warning{Package `textcomp` used to typeset \string\degree\space is\MessageBreak
      not available, switching to "poor man's\MessageBreak
      solution"}%
  }
}
\protected\def\micro{\text{\textmu}}
\protected\def\degree{\text{\textdegree}}

%%%%%%%%%% MAKE TITLE AND TITLE STUFF %%%%%%%%%%
\ifX@c%@c

%%% ABSTRACT
% delete orinigal definition
\let\abstract\relax
\let\endabstract\relax
% abstract collects its contents into \@abstractbody
\NewEnviron{abstract}{\protected@xdef\@abstractbody{\BODY}}
\let\@abstractbody\@empty

%%% DOCUMENT TITLE
\def\title@bracket[#1]#2{%
  \gdef\@shorttitle{#1}%
  \gdef\@title{#2}%
}
\def\title{\@dblarg\title@bracket}
\let\@title\@empty
\let\@shorttitle\@empty

%%% KEYWORDS
\def\keywords#1{\def\@keywords{#1}}
\let\@keywords\@empty

%%% MSC CLASS
\def\MSCclass#1{\def\@MSCclass{#1}}
\let\@MSCclass\@empty

%%% `MAKETITLE` COMMAND
% length \X@mkttlex is used as `ex` in the head, we can modify the exact value in the class argument `@shortenhead=<fraction>`
\newlength\X@mkttlex
\X@mkttlex\dimexpr 1ex-\actapoly@@shortenhead ex\relax
% this is the actual contents of \maketitle
\providecommand\MakeTextUppercase[1]{\MakeUppercase{#1}}
\def\X@formattitle#1{\titlelarge\bfseries\MakeTextUppercase{#1}}
\long\def\X@domkttl{%
  % onecolumn vs twocolumn initial vertical space correction
  \ifX@onecolumn\vspace*{\dimexpr-0.5\dp\strutbox-\baselineskip}\fi
  \vspace*{3.0ex}%
  \vspace*{2.0\X@mkttlex}%
  % typesetting the title; \ifx and \X@mkttlerr are used to produce an error in case \title is missing
  {\centering
    \ifx\@title\@empty
      \X@formattitle{\X@mkttlerr{\string\title}{\textbackslash title}}%
    \else
      \X@formattitle\@title
    \fi
  \par}
  \vspace*{1.0ex}%
  \vspace*{2.0\X@mkttlex}%
  % typeset authors
  {\centering\large\scshape\frenchspacing
    \ifx\@author\@empty\X@mkttlerr{\string\author}{\textbackslash author}\else
      \@author
    \fi
  \par}%
  \vspace*{1.5\X@mkttlex}%
  % institutions are typeset in a customized {itemize} environment
  \ifx\X@print@insts\@empty\else
    {\itemize[itemsep=0.5ex,parsep=0pt,leftmargin=0pt,itemindent=0pt,labelwidth=2ex,labelsep=0pt,topsep=0pt]
    \raggedright\X@print@insts\enditemize
  \par}%
  \fi
  % typeset abstract
  \ifx\@abstractbody\@empty\X@mkttlwrn{`abstract'}{abstract}\else
    \begingroup
      \vspace*{2.0\X@mkttlex}%
      \parskip0pt\relax
      % indent all but first paragraph
      \parindent5ex\relax
      \noindent
      % "Abstract." in small-caps, and a space before the abstract text starts
      {\normalfont\scshape Abstract.} \hskip0.2em plus 0.5em\@abstractbody
      \par
    \endgroup
  \fi
  % keywords, formatting similar to abstract, just without additional space after "Keywords:"
  \ifx\@keywords\@empty\X@mkttlwrn{\string\keywords}{\textbackslash keywords}\else
    \begingroup
      \vspace*{0.5\X@mkttlex}%
      \noindent
      {\normalfont\scshape Keywords:} \@keywords.
    \endgroup
  \fi
  \par
  % MSC, same as keywords
  \ifx\@MSCclass\@empty\else
    \begingroup
      \vspace*{0.5\X@mkttlex}%
      \noindent
      {\scshape AMS Mathematics Subject Classification:} \@MSCclass.
    \endgroup
  \fi
  % here, perform check whether all institutions were used
  \X@check@insts\relax
}
% box to store maketitle contents
\newsavebox\X@ttlbox
% cycle used in case maketitle does not fit in 1 page
\def\X@mkttlrepeat{%
    % \X@xxxcolumn is either \twocolumn or "nothing"
    \X@xxxcolumn[{\@twocolumnfalse
    % check the size of \X@ttlbox
    \ifdim\ht\X@ttlbox>0.97\textheight
    % \X@ttlbox does not fit in the page
      \begingroup
      % infinite \vbadness to avoid errors
      \vbadness10000
      % box0 <- first page of \X@ttlbox
      \setbox0\vsplit\X@ttlbox to 0.94\textheight
      % re-box \X@ttlbox
      \setbox1\vbox{\unvbox\X@ttlbox}%
      \global\setbox\X@ttlbox\vbox{\unvbox1}%
      % box2 <- the content of this page
      \setbox2\vbox to \textheight{%
        % "continued from the previous page"
        \leftline{\kern0.04\textwidth\small\itshape(cont.)}%
        % vertically center the page contents
        \vfill
        \unvbox0
        \vfill
        % "continued on the next page"
        \rightline{\small\itshape(continuted on the next page)\kern0.04\textwidth}%
      }%
      % to avoid bad vbox errors
      \ht2=0.9\textheight
      \box2
      \endgroup
    \else
      % \X@ttlbox fits into the page
      % "continued from the previous page"
      \vbox to 0.03\textheight{\leavevmode\kern0.04\textwidth\small\itshape(cont.)\vfil}%
      % check how much space is left on this page
      \ifdim\ht\X@ttlbox>0.84\textheight
        % if less than ~10%, start the article body on the next page
        \global\let\X@mkttlrepeat\clearpage
      \else
        % if more than ~10%, we can start on this page
        \global\let\X@mkttlrepeat\relax
      \fi
      % show the rest of the header
      \unvbox\X@ttlbox
    \fi
    }]%
  % if this was not the last page then we repeat, if it was then we either do nothing or do \clearpage
  \X@mkttlrepeat
}
% command \maketitle itself
\long\def\maketitle{%
  \begingroup
    \@twocolumnfalse
    % at this moment, we add "corresponding author" as the last institution, and maybe some other stuff
    \X@premkttl    
    % if there's only one author, we want to hide the star, and use different caption for the email
    \ifnum\X@author@cnt=1\relax
      \def\X@corrauthortext{correspondence}%
      \def\X@symbolstar{\relax}%
      \let\X@inst@corrcomma\relax
    \else
      \def\X@corrauthortext{corresponding author}%
    \fi
    % if there's only one institution, hide the symbol "a"
    \ifnum\c@institution=1\relax
      % only the institution number 0="corresponding author" should be shown as "*", other labels are hidden
      \def\symbolnumber##1{\ifnum##1=0\X@symbolstar\else\fi}%
      % no comma after the institution number
      \let\X@inst@comma\relax
      \let\X@inst@corrcomma\relax
    \fi
    % show warning in case of missing \correspondingauthor
    \ifx\X@email\relax
      % so that "corresponding author"=institution 0
      \c@institution-1 %
      \ifX@complain\institution{@}{\color{red}Missing ``\string\correspondingauthor'' before ``\textbackslash maketitle''}\fi%
      % to hide "unused institution `@`" error
      \let\X@check@inst@@\relax
      % typeout warning
      \X@mkttlwrn[]{\string\correspondingauthor}{\textbackslash correspondingauthor}%
    \fi
    % first page has no author or title in the header
    \thispagestyle{plain}%
    % the following code sort-of mimicks the code for \X@mkttlrepeat above
    \X@xxxcolumn[{%
    % FirstPage label
    \label{FirstPage}%
    % set the box containing the maketitle contents
    \global\setbox\X@ttlbox\vbox{%
      % add the article itself as a PDF navigation bookmark
      \pdfbookmark[1]{\X@journal\ \AP@volume(\AP@number):\X@FirstPage--\X@LastPage, \AP@year}{bkmk:mkttl}%
      % we use list with \leftmargin=\rightmargin=0.04\textwidth as the environment to store the title
      % the good thing is that it allows page breaking
      \begin{list}{}{\leftmargin0.04\textwidth\rightmargin0.04\textwidth\topsep0pt\partopsep0pt\itemsep0pt}\item\X@domkttl\end{list}%
      % rule at the end of the maketitle
      {\par\noindent\rule{1.0\textwidth}{0.4pt}\par}%
      \vspace*{2ex}%
      }%
    \ifdim\ht\X@ttlbox>\textheight
      \begingroup
      \vbadness10000
      \setbox0\vsplit\X@ttlbox to 0.97\textheight
      \setbox1\vbox{\unvbox\X@ttlbox}%
      \global\setbox\X@ttlbox\vbox{\unvbox1}%
      \setbox2\vbox to \textheight{%
        \unvbox0
        \vfill
        \rightline{\small\itshape(continuted on the next page)\kern0.04\textwidth}%%%%
      }%
      \ht2=0.9\textheight
      \box2
      \endgroup
    \else
      \ifdim\ht\X@ttlbox>0.87\textheight
        \global\let\X@mkttlrepeat\clearpage
      \else
        \global\let\X@mkttlrepeat\relax
      \fi
      \unvbox\X@ttlbox
    \fi
    }]%
    \X@mkttlrepeat
  \endgroup
  % \maketitle is here, so the error macro \X@nomkttl should be void
  \global\let\X@nomkttl\relax
  % check whether the headings' contents fit into the headings, show warning if not
  \X@chkheadssize
  % disallow unnumbered \section and \subsection
  \gdef\section{\@ifnextchar*{\X@nostarerr\expandafter\section\@gobble}{\ttl@straightclass{section}}}%
  \gdef\subsection{\@ifnextchar*{\X@nostarerr\expandafter\subsection\@gobble}{\ttl@straightclass{subsection}}}%
  % \title, \author, \correspondingauthor, {abstract}, \keywords, \MSCclass and \institution produce an error when placed after \maketitle
  \gdef\title{\X@ttlnoeff{title}\@gobble}%
  \gdef\author{\X@ttlnoeff{author}\@ifnextchar[{\expandafter\@gobbletwo\@gobblebracket}\@gobbletwo}%
  \gdef\correspondingauthor{\X@ttlnoeff{correspondingauthor}\@ifnextchar[{\expandafter\@gobblethree\@gobblebracket}\@gobblethree}%
  \NewEnviron{X@ttlnoeff@abstract}{\X@ttlnoeff{abstract}}%
  \let\abstract\X@ttlnoeff@abstract
  \let\endabstract\endX@ttlenoeff@abstract
  \gdef\keywords{\X@ttlnoeff{keywords}\@gobble}%
  \gdef\MSCclass{\X@ttlnoeff{MSCclass}\@gobble}%
  \gdef\institution{\X@ttlnoeff{institution}\@gobbletwo}%
  % vertical space correction in the case of onecolumn
  \ifX@onecolumn\vspace*{-2.25ex plus -1.25ex minus -.2ex}\fi
}
% helping low-level macros
\def\@gobblethree#1#2#3{}
\def\@gobblebracket[#1]{}
% make sure there is \maketitle before any text
\g@addto@macro{\document}{%
  \everypar\expandafter{\the\everypar\X@nomkttl}%
  \let\section\X@nomkttl
  \let\subsection\X@nomkttl
}
% error macro for title-like commands after \maketitle
\def\X@ttlnoeff#1{\X@Error{The command \expandafter\string\csname#1\endcsname\MessageBreak has no effect after \string\maketitle}{}{}}
% error macro for text before \maketitle
\def\X@nomkttl{\X@Error{Missing \string\maketitle}{}{}}
% "Missing ... before \maketitle" error and warning
\newcommand\X@mkttlerr[3][Missing ``\@tempa'' before ``\textbackslash maketitle'']{\ifX@complain\begingroup
  \def\@tempa{#3}\color{red}#1
  \X@Error{Missing #2\MessageBreak before \string\maketitle}{}{}
\endgroup\fi}
\newcommand\X@mkttlwrn[3][Missing ``\@tempa'' before ``\textbackslash maketitle'']{\ifX@complain\begingroup
  \def\@tempa{#3}\color{red}#1
  \X@Warning{Missing #2\MessageBreak before \string\maketitle}
\endgroup\fi}
% \X@premkttl stores code to be executed at the beginning of \maketitle
\def\X@premkttl{}

%%% INSTITUTIONS
\newcounter{institution}
% \X@check@insts and \X@print@insts store some information about all institutions
\let\X@check@insts\@empty
\let\X@print@insts\@empty
% symbol for "corresponding author"
\def\X@symbolstar{$*$}
\def\X@inst@comma{,\kern0.08333em\relax}
\let\X@inst@corrcomma\X@inst@comma
% special numbering: 0=*, 1-26=a-z, 27-52=A-Z, 53-...=1-...
\def\@aAa#1{%
  \ifnum#1=0\X@symbolstar%
  \else\ifnum#1<27
      \@alph{#1}%
    \else\ifnum#1<53
        \@Alph{\numexpr#1-26\relax}%
      \else
        \@arabic{\numexpr#1-52\relax}%
      \fi
    \fi
  \fi
}
\def\aAa#1{\expandafter\@aAa\csname c@#1\endcsname}% LaTeX
% formatting of institution label
\def\symbolnumber#1{{\normalfont\itshape\@aAa{#1}}}
% this crazy \@@-seperated definition of the macro was the only way to deal with expansions problems later
\def\X@institution#1\@@#2\@@{%
  % test for redefined institution
  \@ifundefined{X@inst@label@#1}{%
    % standard case: then we proceed normally
    % institution++
    \advance\c@institution1\relax
    % globally define the label number for this institution
    \expandafter\xdef\csname X@inst@label@#1\endcsname{\arabic{institution}}%
    % add this institution to the lists of institutions
    \g@addto@macro{\X@check@insts}{\csname X@check@inst@#1\endcsname}%
    \g@addto@macro{\X@print@insts}{\csname X@print@inst@#1\endcsname}%
    % warning macro for unused institution
    \expandafter\gdef\csname X@check@inst@#1\endcsname{\X@Warning{Unused institution `#1`}}%
  }{%
    % re-definition: we do only re-definition of the text (at the end of the macro,
    % after the \@ifundefined is finished)
    % and we just produce a warning
    \X@Warning{Re-defining institution `#1`}%
  }%
  % globally define the correct printing command for this institution
  \expandafter\gdef\csname X@print@inst@#1\endcsname{\X@print@inst{#1}{#2}}%
}
% macro \instution itself just tranfers the arguments to \X@institution
\def\institution#1#2{\X@institution#1\@@#2\@@}
% this is how institutions should look like
\def\X@print@inst#1#2{%
  % for use in the case of undefined institution
  \def\X@instlabel{#1}%
  % no widows and orphans
  \widowpenalty\@M
  \clubpenalty\@M
  % institutions are printed in {itemize}, the label is put into a superscript, the text is smaller and italic
  \item[\smash{\large$^{\text{\symbolnumber{\csname X@inst@label@#1\endcsname}}}$}]{\small\itshape#2}
}
\def\X@use@inst#1{%
  % handle the undefined yet used institutions
  \@ifundefined{X@inst@label@#1}{\begingroup%
    % store the label
    \edef\reserved@a{#1}%
    % print a warning
    \X@Warning{Institution `#1` undefined}{}{}%
    % print the institution anyways, with the error message; this is done by adding it as a real institution
    % this is where we use the very crazy syntax of the macro \X@institution
    \expandafter\X@institution\reserved@a\@@ Undefined institution `\X@instlabel':
      use ``\textbackslash institution\{\X@instlabel\}\{\ldots{}\}'' before ``\textbackslash maketitle''!\@@
  \endgroup}%
  % the institution is used, therefore the check should be void
  \expandafter\global\expandafter\let\csname X@check@inst@#1\endcsname\relax%
  % print the label, the autor printing macro switches to mathmode and superscript, so we needn't care about that
  {\text{\symbolnumber{\csname X@inst@label@#1\endcsname}}}%
}

%%% AUTHORS
% prints one author with his institutions
\def\X@format@author#1#2{
  \leavevmode
  % no line breaks "inside" the author, institutions in superscript
  \hbox{#1$^{\text{%
    % no comma before the first institution of the author
    \gdef\X@corr@comma{%
      \gdef\X@comma{\X@inst@comma}%
      \gdef\X@corrcomma{\X@inst@corrcomma}%
    }%
    \gdef\X@comma{%
      \gdef\X@comma{\X@inst@comma}%
      \gdef\X@corrcomma{\X@inst@corrcomma}%
    }%
    % for cycle through all the institutions of the author
    \def\X@at@sign{@}%
    \@for\@x:=#2\do{\ifx\@x\X@at@sign\X@corrcomma\else\X@comma\fi\X@use@inst{\@x}}%
  }}$}%
}
% macros to store the authors lists
% stores shortened author names (e.g. for the headings)
\let\@shortauthor\@empty
% stores clean list of authors, each one placed in \hbox
\let\@cleanauthor\@empty
% this one stores comma-seperated list of authors as they are printed in \maketitle
\let\@author\@empty
\def\xg@addto@macro#1#2{%
  \begingroup
  \toks@\expandafter{#1#2}%
  \xdef#1{\the\toks@}%
  \endgroup
}
% adds an author
\newcount\X@author@cnt
\def\author@bracket[#1]#2#3{
  % increase number of authors by one
  \advance\X@author@cnt1
  % save author's short name and full name
  \expandafter\def\csname X@author@no@\romannumeral\X@author@cnt\endcsname{{#1}{#2}}%
  % add commas in case this is not the first author and add author into all three lists
  \ifx\@cleanauthor\@empty\else
    \g@addto@macro{\@cleanauthor}{, }%
  \fi
  \g@addto@macro{\@shortauthor}{#1}%
  \ifx\@shortauthor\@empty\else
    \g@addto@macro{\@shortauthor}{, }%
  \fi
  \g@addto@macro{\@cleanauthor}{\relax\hbox{#2}}%
  \ifx\@author\@empty\else
    \g@addto@macro{\@author}{, }%
  \fi
  \begingroup
    \let\'\relax
    \xg@addto@macro{\@author}{\X@format@author{#2}{#3}}%
  \endgroup
}
% corresponding author
\def\correspondingauthor@bracket[#1]#2#3#4{%
  % there is an e-mail
  \xdef\X@email{#4}
  % command to print the e-mail in \maketitle
  \gdef\X@print@email{{\ttfamily #4}}
  % we cannot add the "corresponding author" as an institution now because we want it to be the last one
  \g@addto@macro{\X@premkttl}{\begingroup%
    % so that institution=0
    \c@institution=-1\relax
    % label `@` is reserved for "corresponding author"
    \institution{@}{\upshape\X@corrauthortext: \X@print@email}%
  \endgroup}%
  % we add as a normal author, with the extra institution `@`
  \author[#1]{#2}{#3,@}%
  % only one corresponding author!
  \def\correspondingauthor@bracket[##1]##2##3##4{\X@Error{Only one corresponding author allowed per article}{}}
}
% add information into authors' list
\def\authormetainfo@bracket[#1]#2{
  % add commas in case this is not the first author and add author into all three lists
  \ifx\@cleanauthor\@empty\else
    \g@addto@macro{\@cleanauthor}{ }%
  \fi
  \g@addto@macro{\@shortauthor}{#1}%
  \ifx\@shortauthor\@empty\else
    \g@addto@macro{\@shortauthor}{ }%
  \fi
  \g@addto@macro{\@cleanauthor}{\relax\hbox{#2}}%
  \ifx\@author\@empty\else
    \g@addto@macro{\@author}{ }%
  \fi
  \begingroup
    \let\'\relax
    \xg@addto@macro{\@author}{#2}%
  \endgroup
}
% proper handling of optional arguments
\def\author{\@dblarg\author@bracket}
\def\correspondingauthor{\@dblarg\correspondingauthor@bracket}
\def\authormetainfo{\@dblarg\authormetainfo@bracket}
% short authors default to the list given by \X@shortauthor
\def\@shortauthors{\@shortauthor}
\def\@shortauthors{%
  \ifcase\X@author@cnt\or
    \expandafter\@secondoftwo\X@author@no@i
  \or
    \expandafter\@secondoftwo\X@author@no@i, 
    \expandafter\@secondoftwo\X@author@no@ii
  \or
    \expandafter\@firstoftwo\X@author@no@i, 
    \expandafter\@firstoftwo\X@author@no@ii,
    \expandafter\@firstoftwo\X@author@no@iii
  \or
    \expandafter\@firstoftwo\X@author@no@i, 
    \expandafter\@firstoftwo\X@author@no@ii,
    \expandafter\@firstoftwo\X@author@no@iii,
    \expandafter\@firstoftwo\X@author@no@iv
  \else
    \expandafter\@firstoftwo\X@author@no@i, 
    \expandafter\@firstoftwo\X@author@no@ii,
    \expandafter\@firstoftwo\X@author@no@iii\ \etal
  \fi
}
% however, short authors can be redefined
\def\shortauthors#1{\gdef\@shortauthors{#1}}
% "et al." to be used in short authors
\DeclareRobustCommand\etal{et al.}
% initialization of \X@email
\let\X@email\relax

\else%@c

% in case we are in issue and not in article, \@shortauthors and \@shorttitle are void, so that nothing is displayed in the headings
\let\@shortauthors\relax
\let\@shorttitle\relax

\fi%@c

\def\redbox#1{{\fbox{$\clubsuit$#1$\clubsuit$}}}
\def\redalert#1{{\color{red}#1}}
\def\redboxnames{\redalert{\bfseries\_\!\_\!\_\!\_\!\_\!\_\!\_\!\_}}

%%%%%%%%%% DOCUMENT FORMATTING ELEMENTS %%%%%%%%%%

%%% LISTS
% labels are printed smaller and hence a bit raised
\def\raiselabel#1\endlabel{{\smash{\raise0.15ex\hbox{\normalfont\small#1\ }}}}
% no item seperators
\setlist{noitemsep}
% follows formatting instruction of the {enumitem} package
\setlist[enumerate,1]{label={{\raiselabel(\arabic*.)\endlabel}},
  ref={\bgroup\normalfont(\arabic*.)\egroup},
  itemsep=1ex plus 0.2ex minus 0.1ex, topsep=1ex plus 0.2ex minus 0.1ex,
  listparindent=\parindent, leftmargin=1\parindent, itemindent=1\parindent, labelsep=0pt, labelwidth=2\parindent, labelindent=!}
\setlist[enumerate,2]{label={{\raiselabel(\alph*)\endlabel}},
  ref={\bgroup\normalfont(\theenumi.\alph*.)\egroup},
  topsep=3pt plus 0.5pt minus 0.3pt, itemsep=0ex plus 0.2ex,
  listparindent=\parindent, leftmargin=1\parindent, itemindent=1\parindent, labelsep=0pt, labelwidth=2\parindent, labelindent=!}
\setlist[enumerate,3]{label={{\raiselabel\roman*.\endlabel}},
  ref={\bgroup\normalfont\theenumii.\roman*\egroup},
  topsep=0pt, itemsep=0ex plus 0.2ex,
  listparindent=\parindent, leftmargin=1\parindent, itemindent=0\parindent, labelsep=0pt, labelwidth=2\parindent, labelindent=!}
\setlist[enumerate,4]{label={{\raiselabel\normalsize\scshape\alph*.\endlabel}},
  ref={\bgroup\normalfont\theenumiii.\bgroup\scshape\alph*\egroup\egroup},
  topsep=0pt, itemsep=0ex plus 0.2ex,
  listparindent=\parindent, leftmargin=1\parindent, itemindent=0\parindent, labelsep=0pt, labelwidth=2\parindent, labelindent=!}
\setlist[itemize,1]{label={{\raiselabel$\bullet$\endlabel}}, align=left,
  itemsep=1ex plus 0.2ex minus 0.1ex, topsep=1ex plus 0.2ex minus 0.1ex,
  listparindent=\parindent, leftmargin=1\parindent, itemindent=0\parindent, labelsep=0pt, labelwidth=1\parindent, labelindent=!}
\setlist[itemize,2]{label={{\raiselabel$\triangleright$\endlabel}}, align=left,
  itemsep=0ex plus 0.2ex,
  listparindent=\parindent, leftmargin=1\parindent, itemindent=0\parindent, labelsep=0pt, labelwidth=1\parindent, labelindent=!}
\setlist[itemize,3]{label={{\raiselabel--\endlabel}}, align=left,
  itemsep=0ex plus 0.2ex,
  listparindent=\parindent, leftmargin=1\parindent, itemindent=0\parindent, labelsep=0pt, labelwidth=1\parindent, labelindent=!}
\setlist[description]{font=\bfseries,labelindent=\parindent,style=unboxed}
\ifX@c%@c
% description is defined differently in the article than in the issue
\setlist[description]{font=\bfseries,leftmargin=\parindent,font=\bfseries,labelindent=0\parindent,style=unboxed,
  itemsep=1ex plus 0.2ex minus 0.1ex, topsep=1ex plus 0.2ex minus 0.1ex,
  labelsep=\fontdimen2\font, before=\let\X@orig@item\item\let\item\X@desc@item}
\newcommand\X@desc@item[1][]{\X@orig@item[#1]\leavevmode{\spacefactor3000} \ignorespaces}
\fi%@c

%%% SECTIONS
% section font is smaller than default, small caps, and ragged right (no hyphenation)
% section labels are bold
% \X@chketp checks whether this is the last page and eventually issues proper \enlargethispage to balance the columns
\titleformat{\section}[hang]{\X@chketp\halflarge\scshape\raggedright}{{\upshape\bfseries\thesection.\ }}{0pt}{}
\titleformat{\subsection}[hang]{\X@chketp\halfsmall\scshape\raggedright}{{\upshape\bfseries\thesubsection.\ }}{0pt}{}
\titleformat{\subsubsection}[hang]{\normalsize\scshape\raggedright}{{\upshape\bfseries\thesubsubsection.\ }}{0pt}{}
% tighter section spacing
\titlespacing*{\section}{0pt}{2.5ex plus 1.5ex minus .2ex}{0.8ex plus .2ex minus 0.1ex}
\titlespacing*{\subsection}{0pt}{2.25ex plus 1.25ex minus .2ex}{0.5ex plus .2ex minus 0.1ex}
\titlespacing*{\subsubsection}{0pt}{2ex plus 1ex minus .2ex}{0.3ex plus .2ex minus 0.1ex}
% warning in case of starred \section or \subsection
\def\X@nostarerr{\X@Warning{Sections and subsections have to be numbered}}

%%% FLOATS
% default placement of all floats is `t`
\floatplacement{figure}{t}
\floatplacement{table}{t}
\floatplacement{figure*}{t}
\floatplacement{table*}{t}
% minimum 70% of page height on float pages
\renewcommand{\floatpagefraction}{0.7}
\renewcommand{\dblfloatpagefraction}{0.7}
% 2 floats on top, no on bottom
\setcounter{topnumber}{2}
\setcounter{bottomnumber}{0}
\setcounter{totalnumber}{4}
\setcounter{dbltopnumber}{2}
% up to 90% of floats on a mixed page, at least 7% of text
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.9}
\renewcommand{\dbltopfraction}{0.9}
\renewcommand{\textfraction}{0.07}
% float caption: justification with extra margins on both sides
\DeclareCaptionJustification{myjustify}{\leftskip.8em \rightskip.8em \parfillskip.8em plus 1fil}
% float caption: special format
\DeclareCaptionFormat{myformat}{%
  % float name small caps
  \begingroup\scshape#1#2. \endgroup
  % float caption normal font
  \begingroup#3\par\endgroup
}
\DeclareCaptionFormat{mysubformat}{%
  % float name small caps
  #1#2 %
  % float caption normal font
  \begingroup#3\par\endgroup
}
% plain float style is what we like, with captions at the bottom
\floatstyle{plain}
% restyle already existing floats
\restylefloat{figure}
\restylefloat{table}
% set up the captions: no indent of captions, no seperator (seperator is handled by {myformat})
\captionsetup{format=myformat,indent=0pt,labelsep=none,justification=myjustify,font={small},position=below}
% heavy mid rule for use in booktabs tables
\def\Midrule{\midrule[\heavyrulewidth]}
% float-caption distance
\abovecaptionskip8pt plus 3pt minus 1pt
% graphics allowed: pdf, png, jpg
\DeclareGraphicsExtensions{.pdf,.png,.jpg}
% floats should be centered by default:
\g@addto@macro{\@floatboxreset}{\centering}
% better \IfFileExists
\newif\ifX@fileexists
\long\def\X@IfFileExists#1#2#3{%
  \ifx\input@path\@undefined
    \X@fileexistsfalse
  \else
    \@iffileonpath{#1}{\X@fileexiststrue}{\X@fileexistsfalse}%
  \fi
  \ifX@fileexists
  \else
    \openin\@inputcheck#1 %
      \ifeof\@inputcheck
    \else
      \closein\@inputcheck
      \edef\@filef@und{#1 }%
      \X@fileexiststrue
    \fi
  \fi
  \ifX@fileexists
    \def\reserved@a{#2}%
  \else
    \def\reserved@a{#3}%
  \fi
  \reserved@a
}
\ifactapoly@@bwdir
  \RequirePackage{xpatch}
  \xpatchcmd{\Gin@getbase}{\IfFileExists}{\X@IfFileExists}{}{}
  \graphicspath{{bw/}}
\fi%@bwdir


%%% PAGESTYLE
% mix oldstylenums and smallcaps
\def\@oldstylenums@sc#1{%
  \@tfor\tmpx:=#1\do{%
  \ifcat1\tmpx\relax\oldstylenums\tmpx\else\MakeLowercase{\textsc{\tmpx}}\fi}%
}
% robust way to mix oldstylenums and smallcaps, working in PDF context
\DeclareRobustCommand\osnsc[1]{\texorpdfstring{\expandafter\@oldstylenums@sc\expandafter{#1}}{#1}}
% robust oldstylenums, working in PDF context
\DeclareRobustCommand\osnx[1]{\texorpdfstring{\oldstylenums{#1}}{#1}}
% reset head and foot
\fancyhead{}
\fancyfoot{}
% thicker head rule
\def\headrulewidth{0.667pt}
% default: not to show "PREPRINT" in headings
\let\X@head@preprint\@gobble
% numbers in head and foot should be oldstylenums
\def\X@headnums#1{\expandafter\@oldstylenums@sc\expandafter{#1}}
% check if running heads fit into the headings
% and if title and abstract fit into their limits
\def\X@chkheadssize{%
  % running heads check
  \begingroup
    % box running authors
    \setbox0\hbox{\X@hdLE}%
    % if too wide, show a warning
    \ifdim\wd0>0.667\textwidth
      \X@Warning{Running authors too long, use \string\shortauthors\MessageBreak or optional arguments of
        \string\author\MessageBreak to shorten it}
    \fi
    % box running title
    \setbox0\hbox{\X@hdRO}%
    % if too wide, show a warning
    \ifdim\wd0>0.667\textwidth
      \X@Warning{Running title too long, use optional argument\MessageBreak of \string\title\space to shorten it }%
    \fi
  \endgroup
  % check for whether the title fits in 3 lines
  % everything's grouped to prevent \box0 and \box2 leaking out
  \begingroup
    % the \title text properly boxed for measurements
    \setbox0\vbox{\hsize0.92\textwidth\relax\centering\X@formattitle{\@title}}%
    % a 3-line text boxed in the same way as \title
    \setbox2\vbox{\hsize0.92\textwidth\centering\X@formattitle{M M\newline M M\newline M M\newline M M}}%
    % test if "true title" is >0.95 of "tester title"
    \ifdim\dimexpr\ht0+\dp0\relax>\dimexpr0.95\ht2+0.95\dp2\relax
      % in the case it is, throw a warning
      \X@Warning{The \string\title\space is too long, it ought to fit in\MessageBreak two lines}%
    \fi%dim
  \endgroup
  % check for abstract fitting in the length limits
  % set maximal abstract length
  \newcount\X@maxabstractlength
  \X@maxabstractlength1000\relax
  % everything's grouped, we define some boxes and redefine some commands
  \begingroup
    % ignore paragraphs in abstract
    \let\par\relax
    % real abstract in typewriter font
    \setbox0\hbox{\ttfamily\@abstractbody}%
    % a single character in the same font, should be 1em, but who's sure?
    \setbox2\hbox{\ttfamily A}%
    % we store the maximal allowed width of the box, we allow 10 characters extra to prevent stupid errors
    \@tempdima\dimexpr\the\X@maxabstractlength\wd2+10\wd2\relax
    % test if the abstract does not fit
    \ifdim\wd0>\@tempdima
    % store the width of the "too much" part of the abstract
    \@tempdimb\dimexpr\wd0-\the\X@maxabstractlength\wd2\relax
    % divide by a single character width, dirty trick of dividing a dim by a dim to get a dim :) 
    \divide\@tempdimb by \wd2\relax
    % and store the dimension in a counter to get the integer part properly
    \@tempcnta\@tempdimb
    % display a warning, here we use the counter to tell the user, how much is he over
    \X@Warning{Text of `abstract` is \the\@tempcnta\space characters\MessageBreak too long}%
  \fi%dim
  \endgroup
}
% display line numbers for the left and the right column
\def\X@lnleft{\leavevmode\copy\X@leftlineno}
\def\X@lnright{\leavevmode\copy\X@rightlineno}
% left even: running authors
\def\X@hdLE{{\frenchspacing \@shortauthors}}
% right even: A.P., plus we eventually show the "PREPRINT" text (notice that \llap is an argument of \X@head@preprint !)
\def\X@hdRE{{\scshape \X@journal}\X@head@preprint\llap}
% left odd: volumn and number, plus we eventually show the "PREPRINT" text (notice that \rlap is an argument of \X@head@preprint !)
\def\X@hdLO{\X@head@preprint\rlap{\scshape vol.\ \X@headnums\AP@volume\ \ no.\ \X@headnums\AP@number/\X@headnums\AP@year}}
% right odd: running title
\def\X@hdRO{\@shorttitle}
\def\X@makedoi{10.14311/AP.\AP@year.\AP@volume.\@fourdigits{\X@FirstPage}}
\def\@fourdigits#1{%
  \ifnum#1<1000 0\fi
  \ifnum#1<100 0\fi
  \ifnum#1<10 0\fi
  \ifnum#1<1 0\fi
  #1%
}
% copyright for use in eprint version
\def\X@copyright{\copyright\ Czech Technical University in Prague, \ifx\AP@year\X@XXXX\relax\number\year\else\AP@year\fi}
% weblink for use in eprint version, it is actually pushed down to the "headsep"
\def\X@hdonline@{\llap{\smash{\raise-\baselineskip\hbox{available online at \articlelink}}}}
\let\X@hdonline\X@hdonline@
% doi for use in eprint version (only if present of course), it is actually pushed down to the "headsep"
\ifx\actapoly@@doi\@empty%@doi
  \def\X@hddoi{\rlap{}}
  \def\X@hdDOI{}
\else%@doi
  \def\X@hddoi{\rlap{\smash{\raise-\baselineskip\hbox{\ttfamily\href{http://dx.doi.org/\actapoly@@doi}{{\vphantom{p}doi:\actapoly@@doi}}}}}}
%  \def\X@hdDOI{\leavevmode doi:\actapoly@@doi}
  \def\X@hdDOI{}
\fi%@doi
\ifX@c%@c
  % only for articles and only if not in journal and not "vnyp"ped
  \ifX@injournal\else\ifx\actapoly@@vnyp\@empty
    \ifX@showdate
      % version of "PREPRINT" text showing the current date as well
      % notice that \X@head@preprint takes an argument that should be either \llap or \rlap
      \def\X@head@preprint#1{\smash{\raisebox{\baselineskip}{#1{\sffamily\bfseries
        \MakeTextUppercase{preprint } \number\year-\two@digits\month-\two@digits\day}}}}
      \def\X@hdonline{\llap{\smash{\raise-\baselineskip\hbox{\sffamily\bfseries
        \MakeTextUppercase{preprint } \number\year-\two@digits\month-\two@digits\day}}}}
    \else%@showdate
      % version of "PREPRINT" text showing no date
      % notice that \X@head@preprint takes an argument that should be either \llap or \rlap
      \def\X@head@preprint#1{\smash{\raisebox{\baselineskip}{#1{\sffamily\bfseries\MakeTextUppercase{preprint}}}}}
      \def\X@hdonline{\llap{\smash{\raise-\baselineskip\hbox{\sffamily\bfseries\MakeTextUppercase{preprint}}}}}
    \fi%@showdate
  \fi\fi%@injournal %@vnyp
\fi%@C
% the headings consist of the heading commands \X@hd[LR][OE] and the line number printing command \X@ln(left|right)
% the line printing is activated by [review] class option
\fancyhead[LE]{\X@lnleft\X@hdLE}
\fancyhead[RE]{\X@hdRE\X@lnright}
\fancyhead[LO]{\X@lnleft\X@hdLO}
\fancyhead[RO]{\X@hdRO\X@lnright}
% page numbers in the foot, this is for issue (roman numerals)
\fancyfoot[RO,LE]{\small\thepage}
\ifX@c%@c
  % page numbers in the foot, this is for articles (oldstylenums)
  \fancyfoot[RO,LE]{\small\oldstylenums\thepage}
  % we put \X@XXXX = 0000, the "default year of the article"
  \def\X@XXXX{0000}
  \ifX@eprint
    % in eprint version the first page has different headers
    % first page uses the {plain} pagestyle
    \fancypagestyle{plain}{%
      % reset headers
      \fancyhf{}%
      % left heading: line numbers (\X@lnleft), doi (\X@hddoi) and the bibliography data (rest)
      \fancyhead[L]{\X@lnleft{\small\X@hddoi\emph{\X@journal} \textbf{\AP@volume}%
        ({\scshape\AP@number}):\X@FirstPage--\X@LastPage, \AP@year}}%
      % right heading: copyright (\X@copyright), weblink (\X@hdonline) and line numbers (\X@lnright)
      \fancyhead[R]{{\small\X@copyright\X@hdonline}\X@lnright}%
      % no headrule on the first page
      \def\headrulewidth{0pt}%
      % stadard page number in foot
      \fancyfoot[RO,LE]{\small\oldstylenums\thepage}%
    }
    % currently, all articles are linked to the webpage of AP
    \urldef{\articlelink}\url{http://ojs.cvut.cz/ojs/index.php/ap}
  \else%@eprint
    % in case of [noeprint], the first page differs from the latter ones only by missing running title/authors
    \fancypagestyle{plain}{%
      \fancyhf{}%
      \ps@fancy
      \fancyhead[LE]{\X@lnleft\X@hdDOI}%
      \fancyhead[RE]{\X@hdRE\X@lnright}%
      \fancyhead[LO]{\X@lnleft\X@hdLO}%
      \fancyhead[RO]{\X@hdDOI\X@lnright}%
      \fancyfoot[RO,LE]{\small\oldstylenums\thepage}%
    }
  \fi%@eprint
\fi%@c
% boxes for line numbers
\newsavebox\X@leftlineno
\newsavebox\X@rightlineno
\ifX@review
  % in [review] mode, pseudo-line numbers are shown on left and right
  % we better use a special counter
  \newcounter{X@cnt}
  % the box is of zero size and the environment {picture}(0,0) is used to place everything where it should be
  \setbox\X@leftlineno\hbox to 0pt{{\setlength{\unitlength}{\baselineskip}\begin{picture}(0,0)%
    % reset counter
    \setcounter{X@cnt}{0}
    % the intercept is trial-and-error, the shift is exactly -\baselinsekip
    % the rest is just formatting (typewriter font and \marginparsep seperator)
    \multiput(0,-2.0)(0,-1.01){60}{\hbox to 0pt{\hss\scriptsize\sffamily\stepcounter{X@cnt}\arabic{X@cnt}\kern\marginparsep}}%
    \end{picture}%
  }}
  % very same for right numbers, just starting from 61
  \setbox\X@rightlineno\hbox to 0pt{{\setlength{\unitlength}{\baselineskip}\begin{picture}(0,0)%
    \setcounter{X@cnt}{60}
    \multiput(0,-2.0)(0,-1.01){60}{\hbox to 0pt{\kern\marginparsep\scriptsize\sffamily\stepcounter{X@cnt}
      \ifnum\arabic{X@cnt}<100 \phantom{0}\fi\arabic{X@cnt}\hss}}%
    \end{picture}%
  }}
\else%@review
  % the boxes are empty in case of [noreview]
  \setbox\X@leftlineno\hbox{}
  \setbox\X@rightlineno\hbox{}
\fi%@review
% {fancy} is the default pagestyle
\pagestyle{fancy}
% code for manual balancing of the columns on the last page
\ifx\actapoly@@shortenlast\@empty%@shortenlast
  % if no balancing is specified, do nothing
  \let\X@etp\relax
\else%@shortenlast
  % if it is specified, 1) make sure that the code is run only once, with issuing \raggedbottom in whatever follows
  % 2) enlarge the column by the specified negative fraction of \textheight
  \def\X@etp{\global\let\X@etp\raggedbottom\raggedbottom\enlargethispage{-\actapoly@@shortenlast\textheight}}
  % tweak into \@makecol to make footnotes appear at the very bottom in the shortened column.
  \gdef\@makecol{%
    \ifvoid\footins
      \setbox\@outputbox\box\@cclv
    \else
      \setbox\@outputbox\vbox{%
        \boxmaxdepth \@maxdepth
        \unvbox\@cclv
        % here, we add 4 lines so that \box\@kludgeins is inserted before \box\footins
        \ifvbox\@kludgeins
          \box\@kludgeins
          \vfill
        \fi
        \vskip \skip\footins
        \color@begingroup
          \normalcolor
          \footnoterule
          \unvbox\footins
        \color@endgroup
       }%
    \fi
    \let\@elt\relax
    \xdef\@freelist{\@freelist\@midlist}%
    \global\let\@midlist\@empty
    \@combinefloats
    \ifvbox\@kludgeins
      \@makespecialcolbox
    \else
      \setbox\@outputbox\vbox to\@colht {%
        \@texttop
        \dimen@\dp\@outputbox
        \unvbox\@outputbox
        \vskip-\dimen@
        \@textbottom
       }%
    \fi
    \global\maxdepth\@maxdepth
  }
\fi%@shortenlast
\ifX@c%@c
  % in article, \X@chketp checks whether we're on the last page and if so, issues \X@etp
  % the command is used in all sectioning commands and in every \bibitem; this should be enough
  % to make sure it's issued at least once in the first column of the last page
  \def\X@chketp{\ifnum\numexpr\thepage=\X@LastPage\relax\X@etp\fi}
  \def\checklastpage{\X@chketp}
\else%@c
  % in issue, we do nothing
  \let\X@chketp\relax
\fi%@c

%%% BIBLIOGRAPHY, ACKNOWLEDGEMENTS AND NOMENCLATURE

\ifX@c%@c
\@ifpackageloaded{natbib}{}{
\def\thebibliography#1{%
  % issue \subsection* this way since we otherwise forbid it
  \ttl@straightclass{subsection}*{\refname}%
  % add References to PDF navigation
  \pdfbookmark[1]{References}{bkmk:references}%
  % font size
  \loosesmall
  % this mimicks the original definition, with some modifications
  \list {\relax}{%
    \labelsep0pt
    % almost raggedright
    \@rightskip0pt plus 160pt
    \rightskip\@rightskip
    % first line of item to the very left, other lines indented
    \leftmargin0.66666em
    \itemindent-0.66666em
    % small itemsep
    \parsep\z@
    \itemsep1ex \@plus 1ex \@minus 0.5ex
    % the following is directly from LaTeX
    \def\newblock{\hskip.11em\@plus.33em\@minus.07em}
    \@openbib@code
    \usecounter {enumiv}%
    \let\p@enumiv\@empty
    \renewcommand\theenumiv{\@arabic\c@enumiv}%
  }
  % reasonably forbid widows and orphans
  \sloppy \clubpenalty 4000%
  \@clubpenalty \clubpenalty
  \widowpenalty 4000%
  % dots do not terminate sentences
  \sfcode `\.\@m
  % code specific for case without natbib
  \X@bibpreamble
}
}%@ifpackageloaded
\def\X@bibpreamble{%
  % item should behave differently
  % first, store original \@item
  \let\X@item\@item
  \def\X@gobblebracket[##1]{}%
  \def\@item{%
    % check for the first column of the last page
    \X@chketp
    % start the item with no label
    \X@item[\relax]\relax
    % print the label
    [\@arabic\c@enumiv]%
    % space after the label
    \kern0.33333em\ignorespaces
    % if the item had an optional argument, ignore it
    \expandafter\ignorespaces\@ifnextchar[\X@gobblebracket\relax
  }%
  % slightly increased \linepenalty
  \linepenalty400
}
% {natbib} version
\def\X@natbibpreamble{%
  % first, store original \item (not \@item !!!)
  \let\X@item\item
  \def\item[##1]{%
    % check for the first column of the last page
    \X@chketp
    % start the item with no label
    \X@item[\relax]\leavevmode
    % print the item parameter, but ignore \hfil that is there
    \ifX@onecolumn{\let\hfil\relax\hbox to \bibhang{\hfill##1\kern0.33333em}}\else{\let\hfil\relax##1\kern0.33333em}\fi
    % add space
    \ignorespaces
  }%
  \ifX@onecolumn
    \def\NAT@bibsetup##1{%
      \settowidth{\bibhang}{[##1]\kern0.66666em}%
      \settowidth{\leftmargin}{[##1]\kern0.6666em}%
%      \setlength{\leftmargin}{\bibhang}%
      \setlength{\itemindent}{-\leftmargin}%
      \setlength{\itemsep}{\bibsep}%
      \setlength{\parsep}{\z@}%
    }%
  \fi%@onecolumn
  % no indent
  \parindent\z@
  % explain the {natbib} package which \@bibsetup to use
  \let\@bibsetup\NAT@bibsetup
  % store original \bibitem and redefine
  \let\X@@bibitem\bibitem
  \def\X@bibitem[##1]##2{\X@@bibitem[##1]{##2}\ignorespaces}
  \def\bibitem{%
    \stepcounter{X@bibstep}%
    % if there's an optional argument, let's hope it's in the proper format
    \@ifnextchar[\X@bibitem
    % if there's not, let's make one in the form A(1001)
    {\X@bibitem[A(\the\c@X@bibstep)]}%
  }
  % slightly increased \linepenalty
  \linepenalty400
}
% eprint command is a variant of \url with an optional argument
% without optional argument \url{#1}, with optional argument \href{#2}{\nolinkurl{#1}}
\def\X@eprint@br[#1]#2{\href{#2}{\nolinkurl{#1}}}
\def\eprint{\@ifnextchar[\X@eprint@br\url}
\def\arXiv#1{\eprint[arXiv:#1]{http://arxiv.org/abs/#1}}
% DOI typesetting command -- the same one as in actapoly*.bst
\def\bibdoi{\begingroup\def\do##1{\catcode
  `##112\relax}\do$\do\\\do\_\do\%\do\^\expandafter\endgroup\@bibdoi}
\def\@bibdoi#1{\href{http://dx.doi.org/#1}{\textsc{doi}:#1}}

\AtBeginDocument{\@ifpackageloaded{natbib}{
  \newcounter{X@bibstep}
  \setcounter{X@bibstep}{2000}
  \def\bibsection{%
    \ttl@straightclass{subsection}*{\refname}%
    \pdfbookmark[1]{References}{bkmk:references}%
  }
  \bibhang0.66666em
  \bibsep1ex \@plus 1ex \@minus 0.5ex
  \def\bibfont{\normalfont\small\raggedright}%
  \let\bibpreamble\X@natbibpreamble
}{}}

% forbid \thanks
\def\thanks{\X@Error{Command `\string\thanks` invalid, use the\MessageBreak environment `acknowledgements`}}
\newenvironment{acknowledgements}{%
  % issue \subsection* this way since we otherwise forbid it
  \ttl@straightclass{subsection}*{Acknowledgements}%
  % add Acknowledgements to PDF navigation
  \pdfbookmark[1]{Acknowledgements}{bkmk:acknowledgements}%
  % font size
  \loosesmall
  % be a bit "sloppy"
  \tolerance 5000\emergencystretch 1em
}{%
  % end acknowledgements and restore fontsize (probably not necessary)
  \par\normalsize
}
\long\def\THANKS#1{\begin{acknowledgements}#1\end{acknowledgements}}

%inside this group, we do catcode manipulations of ^ and .
\begingroup
% we need to store the original meaning of ^
\gdef\X@nomencl@sup#1{^{\smash[t]{#1}\vphantom{x}}}
\gdef\X@nomencl@supmin-#1{\X@nomencl@sup{-#1}}
\catcode`\.=\active
\catcode`\^=\active
\catcode`\$=\active
% this changes the catcodes, inside a group
\gdef\X@nomencl@cat{%
  \begingroup
  \catcode`\.=\active
  \catcode`\^=\active
  \catcode`\$=\active
}
\gdef\X@nomencl@icat{%
  \begingroup
  \catcode`\$=\active 
}
% this redefines the commands and actives, it's seperate from \X@nomencl@cat
% so that we can apply it when the argument is processed, not when it's read
\gdef\X@nomencl@catdef{%
  % we don't like dots, we prefer small spaces
  \def.{\,}
  \def\cdot{\,}%
  % if ^ is followed by - we basically just change ^-#1 to ^{-#1}
  \def^{\@ifnextchar-\X@nomencl@supmin\X@nomencl@sup}%
  \def$##1${\ensuremath{\mathnormal{##1}}}
}
\gdef\X@nomencl@icatdef{%
  \def$##1${\ensuremath{\mathnormal{##1}}}
}
% end of the group started by \X@nomencl@cat
\gdef\X@nomencl@catend{
  \endgroup
}
\gdef\X@nomencl@icatend{
  \endgroup
}
\endgroup
\newenvironment{nomenclature}[1][List of symbols]{
  \ttl@straightclass{subsection}*{#1}
  \pdfbookmark[1]{List of symbols}{bkmk:nomenclature}
  \loosesmall
  \begin{description}[itemsep=0.4ex plus 0.1ex minus 0.05ex]
  \let\X@nomencl@item\item
  % define new item, first looking for [
  \def\item{\@ifnextchar[{\X@nomencl@cat\X@nomencl@ia}{\X@nomencl@cat\X@nomencl@ia[]}}
  % if we recieve [- we change - into \text{--}
  \def\X@nomencl@ia[{\@ifnextchar-{\X@nomencl@ib[\text{--}\@gobble}{\X@nomencl@ib[}}
  \def\X@nomencl@ic[##1]##2{%
    \X@nomencl@icatend
    \X@nomencl@id[##1]{##2}%
  }
  % this measures the symbol, if it's too narrow, we increase \wd0 to add some horizontal space
  % the goal is to make the "river" more smooth
  \def\X@nomencl@ii##1{%
    \setbox0\hbox{${\X@nomencl@icatdef##1}$}%
    \ifdim\wd0<0.5em\relax
      \wd0 0.9em\relax
    \else\ifdim\wd0<1.3em\relax
      \wd0\dimexpr0.5\wd0+0.65em\relax
    \fi\fi
    % we use the optional argument of {description}'s \item for the symbol
    \X@nomencl@item[\box0]%
    \ignorespaces
  }
  % the reading of the 1st (optional) argument of \item is seperate so that we can read
  % it with catcodes changed
  \def\X@nomencl@ib[##1]{%
    \X@nomencl@catend
    \X@nomencl@icat
    \X@nomencl@ic[##1]%
  }
  % finally, here comes the variant with the optional argument
  \def\X@nomencl@id[##1]##2##3{%
    \X@nomencl@ii{##2}%
    ##3%
    \ifx&##1&\else
      \unskip\allowbreak
      $\;\;\left[\mathrm{\X@nomencl@catdef##1}\right]$%
    \fi
    \par
  }
}{
  \end{description}
}
\fi%@c

\endinput
