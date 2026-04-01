/*************************************************/
/* KPI Report — Fleet Fuel Efficiency            */
/* Summarises average city MPG by vehicle type   */
/* and flags categories below a threshold.       */
/*************************************************/
%let threshold = 20;

proc means data=sashelp.cars mean noprint;
    var MPG_City;
    class Type;
    output out=work.kpi mean=avg_mpg;
run;

data work.kpi_flagged;
    set work.kpi;
    where _TYPE_ = 1;
    length flag $3;
    if avg_mpg < &threshold then flag = 'LOW';
    else flag = 'OK';
run;

title "Fleet Fuel Efficiency KPI — Avg City MPG by Type";
footnote "Threshold: &threshold MPG";

proc report data=work.kpi_flagged;
    columns Type avg_mpg flag;
    define Type     / group   'Vehicle Type';
    define avg_mpg  / mean    'Avg MPG (City)' format=6.1;
    define flag     / display 'Status';
run;

title; footnote;