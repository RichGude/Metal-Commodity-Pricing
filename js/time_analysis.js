// Create Price Graph
am4core.ready(function() {

    // Themes begin
    am4core.useTheme(am4themes_animated);
    // Themes end


    // Create Color Scheme colors for ACF and PACF plots
    var acfColor = am4core.color("#d36161");        // red
    var pacfColor = am4core.color("#616cd3");       // blue 
    var contColor = am4core.color("#b03fa8");       // purple

    var aluminumColor = am4core.color("#ff8726");   // orange
    var copperColor = am4core.color("#d21a1a");     // red 
    var steelColor = am4core.color("#45d21a");      // green
    var nickelColor = am4core.color("#1c5fe5");     // blue
    var zincColor = am4core.color("#c100e8");       // magenta
    var cpiColor = am4core.color("#7132a8");        // violet

    // ****************************NEW GRAPH********************************************************************//

    /* PLOT OF CONSUMER PRICE INDEX - XY CHART
        A simple graph to show the steady rise of CPI before showing real price change. */

    // Create chart instance
    var cpiChart = am4core.create("cpi_div", am4charts.XYChart);
    cpiChart.dataSource.url = 'PriceData/cpi.json';
    cpiChart.dataSource.parser = new am4core.JSONParser();
    cpiChart.responsive.enabled = true;

    // Create chart axes
    var dateAxis = cpiChart.xAxes.push(new am4charts.DateAxis());
    dateAxis.dataFields.category = "Date";
    dateAxis.title.text = "Year";
    var cpiAxis = cpiChart.yAxes.push(new am4charts.ValueAxis());
    cpiAxis.title.text = "CPI";

    // Create BarChart series for CPI values
    var cpiSeries = cpiChart.series.push(new am4charts.LineSeries());
    cpiSeries.dataFields.valueY = "CPI";
    cpiSeries.dataFields.dateX = "Date";
    cpiSeries.yAxis = cpiAxis;
    cpiSeries.tooltipText = "{Date.formatDate('yyyy-MMM')}\n[bold]CPI:[/] {valueY.formatNumber('#.###')}";
    cpiSeries.name = "CPI";
    cpiSeries.stroke = cpiColor;
    cpiSeries.tooltip.getFillFromObject = false;
    cpiSeries.tooltip.background.fill = cpiColor;
    cpiSeries.strokeWidth = 3;
    cpiSeries.strokeOpacity = 0.4;

    // Add cursor
    cpiChart.cursor = new am4charts.XYCursor();
    cpiChart.cursor.xAxis = dateAxis;

 // ****************************NEW GRAPH********************************************************************//

    /* PLOT OF REAL PRICE CHANGE - XY CHART
        A line chart showing the change in real price compared to the present-day price of each commodity. */

    // Create chart instance
    var chart = am4core.create("price_div", am4charts.XYChart);
    // Load price data saved in .json format
    chart.dataSource.url = 'PriceData/cpi.json';
    chart.dataSource.parser = new am4core.JSONParser();
    chart.responsive.enabled = true;

    // Create chart axes
    var dateAxis = chart.xAxes.push(new am4charts.DateAxis());
    dateAxis.dataFields.category = "Date";
    dateAxis.title.text = "Year";
    var linValAxis = chart.yAxes.push(new am4charts.ValueAxis());
    linValAxis.logarithmic = true;
    linValAxis.title.text = "Metal Price (US$/metric ton)";


    // Count number of columns
    var cols = ['Aluminum', 'Copper', 'IronOre', 'Nickel', 'Zinc'];
    var colorList = {'Aluminum': aluminumColor, 'Copper': copperColor, 'IronOre': steelColor, 'Nickel': nickelColor, 'Zinc': zincColor};

    // Create series for current year price of each commodity
    function createYearSeries(name) {
        var series = chart.series.push(new am4charts.LineSeries());
        series.dataFields.valueY = name;
        series.dataFields.dateX = "Date";
        series.name = name;
        series.stroke = colorList[name];
        series.strokeDasharray = "5,2";
        series.strokeWidth = 1.5;
        series.strokeOpacity = 0.3;
        series.hiddenInLegend = true;

        var segment = series.segments.template;
        segment.interactionsEnabled = true;

        var hoverState = segment.states.create("hover");
        hoverState.properties.strokeWidth = 4;

        var dimmed = segment.states.create("dimmed");
        dimmed.properties.stroke = am4core.color("#dadada");

        //Define hover-over events
        segment.events.on("over", function(event) {
            processOver(event.target.parent.parent.parent);
        });
        segment.events.on("out", function(event) {
            processOut(event.target.parent.parent.parent);
        });
        return series;
    }

    // Create series ('contYear' is 'false' for current day prices, 'true' for constant-1990 prices)
    function createConstSeries(name) {
        var series = chart.series.push(new am4charts.LineSeries());
        // Grab 1990 data from json file but use the same metal name
        series.dataFields.valueY = name + '90';
        series.dataFields.dateX = "Date";
        series.name = name;   // Need to change name variable for calling purposes...
        // Mouse over tooltip text
        series.tooltipText = "[bold]{Date.formatDate('yyyy-MMM')}[/]\n[bold]{name}[/]: {valueY.formatNumber('$#.##')}";
        series.tooltip.getFillFromObject = false;
        series.tooltip.background.fill = colorList[name];
        series.stroke = colorList[name];
        series.strokeWidth = 2;

        var segment = series.segments.template;
        segment.interactionsEnabled = true;

        var hoverState = segment.states.create("hover");
        hoverState.properties.strokeWidth = 4;

        var dimmed = segment.states.create("dimmed");
        dimmed.properties.stroke = am4core.color("#dadada");

        // Define hover-over events
        segment.events.on("over", function(event) {
            processOver(event.target.parent.parent.parent);
        });
        segment.events.on("out", function(event) {
            processOut(event.target.parent.parent.parent);
        });
        return series;
    }
    
    // Create a AM Chart series for each metal for current prices
    cols.forEach(createYearSeries);
    // Create a AM Chart series for each metal for current prices
    cols.forEach(createConstSeries);

    chart.legend = new am4charts.Legend();
    chart.legend.position = "bottom";
    chart.legend.scrollable = true;
    chart.legend.itemContainers.template.events.on("over", function(event) {
        processOver(event.target.dataItem.dataContext);
    })

    chart.legend.itemContainers.template.events.on("out", function(event) {
        processOut(event.target.dataItem.dataContext);
    })

    // Add cursor
    chart.cursor = new am4charts.XYCursor();
    chart.cursor.xAxis = dateAxis;

    function processOver(hoveredSeries) {
        hoveredSeries.toFront();

        hoveredSeries.segments.each(function(segment) {
            segment.setState("hover");})

        chart.series.each(function(series) {
            if (series != hoveredSeries) {
            series.segments.each(function(segment) {
                segment.setState("dimmed");})
            series.bulletsContainer.setState("dimmed");}
        });
    }

 // ****************************NEW GRAPH********************************************************************//

    // Create a Radar chart for displaying zero-difference ADF values
    var zDiffchart = am4core.create("zero_dif_div", am4charts.RadarChart);
    zDiffchart.hiddenState.properties.opacity = 0; // this creates initial fade-in
    // Load adf data saved in .json format
    zDiffchart.dataSource.url = 'PriceData/adf.json';
    zDiffchart.dataSource.parser = new am4core.JSONParser();

    var categoryAxis = zDiffchart.xAxes.push(new am4charts.CategoryAxis());
    categoryAxis.dataFields.category = "Metal";
    categoryAxis.renderer.labels.template.location = 0.2;
    categoryAxis.renderer.tooltipLocation = 0.5;

    var valueAxis = zDiffchart.yAxes.push(new am4charts.ValueAxis());
    valueAxis.tooltip.disabled = true;
    valueAxis.renderer.labels.template.horizontalCenter = "left";
    valueAxis.min = 0;
    valueAxis.max = 0.10;

    // Highlight the area of the graph below 0.05 as green and anything above as red (for significant and insignificant)
    valueAxis.events.on("datavalidated", function (ev) {
        var axis = ev.target;

        var greenRange = axis.axisRanges.create();
        greenRange.value = 0;
        greenRange.endValue = 0.05;
        greenRange.axisFill.fill = am4core.color("#00c70a");
        greenRange.axisFill.fillOpacity = 0.2;
        greenRange.grid.strokeOpacity = 0;

        var redRange = axis.axisRanges.create();
        redRange.value = 0.05;
        redRange.endValue = 0.20;
        redRange.axisFill.fill = am4core.color("#c70700");
        redRange.axisFill.fillOpacity = 0.2;
        redRange.grid.strokeOpacity = 0;
    })

    var zeroDiff = zDiffchart.series.push(new am4charts.RadarColumnSeries());
    zeroDiff.columns.template.tooltipText = "[bold]{Metal} p-value:[/]\n{valueY.value.formatNumber('#.###')}";
    zeroDiff.stroke = am4core.color("#590055")
    zeroDiff.columns.template.tooltipY = am4core.percent(0);
    zeroDiff.columns.template.width = am4core.percent(80);
    zeroDiff.name = "ADF Probability";
    zeroDiff.dataFields.categoryX = "Metal";
    zeroDiff.dataFields.valueY = "zDiff";

    zDiffchart.cursor = new am4charts.RadarCursor();
    zDiffchart.cursor.xAxis = categoryAxis;
    zDiffchart.cursor.fullWidthXLine = true;
    zDiffchart.cursor.lineX.strokeOpacity = 0;
    zDiffchart.cursor.lineX.fillOpacity = 0.1;
    zDiffchart.cursor.lineX.fill = am4core.color("#000000");

 // ****************************NEW GRAPH********************************************************************//

    // Create first difference line chart instance
    var firstDiffChart = am4core.create("diff_line_div", am4charts.XYChart);
    // Load price data saved in .json format
    firstDiffChart.dataSource.url = 'PriceData/cpi.json';
    firstDiffChart.dataSource.parser = new am4core.JSONParser();
    firstDiffChart.responsive.enabled = true;

    // Create chart axes
    var dateAxis = firstDiffChart.xAxes.push(new am4charts.DateAxis());
    dateAxis.dataFields.category = "Date";
    dateAxis.title.text = "Year";
    var linValAxis = firstDiffChart.yAxes.push(new am4charts.ValueAxis());
    linValAxis.title.text = "Metal Price (US$/metric ton)";
    linValAxis.min = -10000;
    linValAxis.max = 10000;

    // AMChart supports calculating first difference series from original series very easily:
    function createDiffSeries(name) {
        var series = firstDiffChart.series.push(new am4charts.LineSeries())
        series.dataFields.valueY = name;
        series.dataFields.valueYShow = "previousChange";
        series.dataFields.dateX = "Date";
        series.name = name;
        series.stroke = colorList[name];
        series.fill = colorList[name];
        series.tooltipText = "[bold]{Date.formatDate('yyyy-MMM')}[/]\n[bold]{name}[/]: {valueY.previousChange.formatNumber('$#.##')}";
        series.tooltip.getFillFromObject = false;
        series.tooltip.background.fill = colorList[name];
        series.stroke = colorList[name];
        series.strokeWidth = 2;
        
        var segment = series.segments.template;
        segment.interactionsEnabled = true;

        var hoverState = segment.states.create("hover");
        hoverState.properties.strokeWidth = 4;

        var dimmed = segment.states.create("dimmed");
        dimmed.properties.stroke = am4core.color("#dadada");

        // Define hover-over events
        segment.events.on("over", function(event) {
            processOver(event.target.parent.parent.parent);
        });
        segment.events.on("out", function(event) {
            processOut(event.target.parent.parent.parent);
        });

        return series;
    }
    
    // Create a AM Chart series for each metal for current prices
    cols.forEach(createDiffSeries);

    firstDiffChart.legend = new am4charts.Legend();
    firstDiffChart.legend.position = "bottom";
    firstDiffChart.legend.scrollable = true;
    firstDiffChart.legend.itemContainers.template.events.on("over", function(event) {
        processOver(event.target.dataItem.dataContext);
    })

    firstDiffChart.legend.itemContainers.template.events.on("out", function(event) {
        processOut(event.target.dataItem.dataContext);
    })

    // Add cursor
    firstDiffChart.cursor = new am4charts.XYCursor();
    firstDiffChart.cursor.xAxis = dateAxis;

    function processOver(hoveredSeries) {
        hoveredSeries.toFront();

        hoveredSeries.segments.each(function(segment) {
            segment.setState("hover");})

            firstDiffChart.series.each(function(series) {
            if (series != hoveredSeries) {
            series.segments.each(function(segment) {
                segment.setState("dimmed");})
            series.bulletsContainer.setState("dimmed");}
        });
    }


 // ****************************NEW GRAPH********************************************************************//

    // Create a Radar chart for displaying first-difference ADF values
    var fDiffchart = am4core.create("first_dif_div", am4charts.RadarChart);
    fDiffchart.hiddenState.properties.opacity = 0; // this creates initial fade-in
    // Load adf data saved in .json format
    fDiffchart.dataSource.url = 'PriceData/adf.json';
    fDiffchart.dataSource.parser = new am4core.JSONParser();

    var categoryAxis = fDiffchart.xAxes.push(new am4charts.CategoryAxis());
    categoryAxis.dataFields.category = "Metal";
    categoryAxis.renderer.labels.template.location = 0.5;
    categoryAxis.renderer.tooltipLocation = 0.5;

    var valueAxis = fDiffchart.yAxes.push(new am4charts.ValueAxis());
    valueAxis.tooltip.disabled = true;
    valueAxis.renderer.labels.template.horizontalCenter = "left";
    valueAxis.min = 0;
    valueAxis.max = 0.10;

    // Highlight the area of the graph below 0.05 as green and anything above as red (for significant and insignificant)
    valueAxis.events.on("datavalidated", function (ev) {
        var axis = ev.target;

        var greenRange = axis.axisRanges.create();
        greenRange.value = 0;
        greenRange.endValue = 0.05;
        greenRange.axisFill.fill = am4core.color("#00c70a");
        greenRange.axisFill.fillOpacity = 0.2;
        greenRange.grid.strokeOpacity = 0;

        var redRange = axis.axisRanges.create();
        redRange.value = 0.05;
        redRange.endValue = 0.20;
        redRange.axisFill.fill = am4core.color("#c70700");
        redRange.axisFill.fillOpacity = 0.2;
        redRange.grid.strokeOpacity = 0;
    })

    var fDiff = fDiffchart.series.push(new am4charts.RadarColumnSeries());
    fDiff.columns.template.tooltipText = "[bold]{Metal} p-value:[/]\n{valueY.value.formatNumber('#.###')}";
    fDiff.columns.template.width = am4core.percent(80);
    fDiff.stroke = am4core.color("#590055")
    fDiff.name = "ADF Probability";
    fDiff.dataFields.categoryX = "Metal";
    fDiff.dataFields.valueY = "fDiff";

    fDiffchart.cursor = new am4charts.RadarCursor();
    fDiffchart.cursor.xAxis = categoryAxis;
    fDiffchart.cursor.fullWidthXLine = true;
    fDiffchart.cursor.lineX.strokeOpacity = 0;
    fDiffchart.cursor.lineX.fillOpacity = 0.1;
    fDiffchart.cursor.lineX.fill = am4core.color("#000000");

 // ****************************NEW GRAPH********************************************************************//

    /* PLOT OF ZERO-DIFFERENCE ALUMINUM PRICE ACF - LOLLIPOP CHART
        A simple, stem plot of ACF values out to 12 lags (~1 year). */

    var zeroACFchart = am4core.create("zeroACF_div", am4charts.XYChart);
    // Load adf data saved in .json format
    zeroACFchart.dataSource.url = 'TimeSeries/acfZero.json';
    zeroACFchart.dataSource.parser = new am4core.JSONParser();

    // Create X-Axis as a category axis for lollipop chart
    var categoryAxis = zeroACFchart.xAxes.push(new am4charts.CategoryAxis());
    categoryAxis.dataFields.category = "Lag";
    categoryAxis.renderer.minGridDistance = 60;
    categoryAxis.title.text = 'Lag';

    // Create Y-Axis
    var valueAxis = zeroACFchart.yAxes.push(new am4charts.ValueAxis());
    valueAxis.tooltip.disabled = true;
    valueAxis.renderer.minGridDistance = 40;
    valueAxis.renderer.axisFills.template.disabled = true;
    valueAxis.title.text = '[bold]ACF Value[/]';
    valueAxis.min = 0;
    valueAxis.max = 1;
    valueAxis.extraMax= 0.1;

    // Highlight the area of the graph below 0.2 as red (for insignificant)
    valueAxis.events.on("datavalidated", function (ev) {
        var axis = ev.target;

        var redRange = axis.axisRanges.create();
        redRange.value = -0.20;
        redRange.endValue = 0.20;
        redRange.axisFill.fill = am4core.color("#c70700");
        redRange.axisFill.fillOpacity = 0.2;
        redRange.grid.strokeOpacity = 0;
    })

    // Fill Chart
    var series = zeroACFchart.series.push(new am4charts.ColumnSeries());
    series.dataFields.categoryX = "Lag";
    series.dataFields.valueY = "ACF";

    // Set Tooltip Values and background color
    series.tooltipText = "[bold]Lag: {Lag}[/]\n[bold]ACF:[/]: {valueY.formatNumber('#.###')}";
    series.tooltip.getFillFromObject = false;
    series.tooltip.background.fill = acfColor;

    series.sequencedInterpolation = true;
    series.stroke = acfColor;
    series.fillOpacity = 0;
    series.strokeOpacity = 1;
    series.strokeDashArray = "1,3";
    series.columns.template.width = 0.02;
    series.tooltip.pointerOrientation = "horizontal";

    var bullet = series.bullets.create(am4charts.CircleBullet);
    bullet.fill = acfColor;

    zeroACFchart.cursor = new am4charts.XYCursor();

 // ****************************NEW GRAPH********************************************************************//

    /* PLOT OF ZERO-DIFFERENCE ALUMINUM PRICE PACF - LOLLIPOP CHART
        A simple, stem plot of PACF values out to 12 lags (~1 year). */

    var zeroPACFchart = am4core.create("zeroPACF_div", am4charts.XYChart);
    // Load adf data saved in .json format
    zeroPACFchart.dataSource.url = 'TimeSeries/acfZero.json';
    zeroPACFchart.dataSource.parser = new am4core.JSONParser();

    // Create X-Axis as a category axis for lollipop chart
    var categoryAxis = zeroPACFchart.xAxes.push(new am4charts.CategoryAxis());
    categoryAxis.dataFields.category = "Lag";
    categoryAxis.renderer.minGridDistance = 60;
    categoryAxis.title.text = 'Lag';

    // Create Y-Axis
    var valueAxis = zeroPACFchart.yAxes.push(new am4charts.ValueAxis());
    valueAxis.tooltip.disabled = true;
    valueAxis.renderer.minGridDistance = 40;
    valueAxis.renderer.axisFills.template.disabled = true;
    valueAxis.title.text = '[bold]Partial ACF Value[/]';
    valueAxis.min = -0.25;
    valueAxis.max = 1;
    valueAxis.extraMax= 0.1;

    // Highlight the area of the graph below 0.2 as red (for insignificant)
    valueAxis.events.on("datavalidated", function (ev) {
        var axis = ev.target;

        var redRange = axis.axisRanges.create();
        redRange.value = -0.20;
        redRange.endValue = 0.20;
        redRange.axisFill.fill = am4core.color("#c70700");
        redRange.axisFill.fillOpacity = 0.2;
        redRange.grid.strokeOpacity = 0;
    })

    // Fill Chart
    var series = zeroPACFchart.series.push(new am4charts.ColumnSeries());
    series.dataFields.categoryX = "Lag";
    series.dataFields.valueY = "PACF";

    // Set Tooltip Values and background color
    series.tooltipText = "[bold]Lag: {Lag}[/]\n[bold]PACF:[/]: {valueY.formatNumber('#.###')}";
    series.tooltip.getFillFromObject = false;
    series.tooltip.background.fill = pacfColor;

    series.sequencedInterpolation = true;
    series.stroke = pacfColor;
    series.fillOpacity = 0;
    series.strokeOpacity = 1;
    series.strokeDashArray = "1,3";
    series.columns.template.width = 0.02;
    series.tooltip.pointerOrientation = "horizontal";

    var bullet = series.bullets.create(am4charts.CircleBullet);
    bullet.fill = pacfColor;

    zeroPACFchart.cursor = new am4charts.XYCursor();

 // ****************************NEW GRAPH********************************************************************//

    /* PLOT OF FIRST-DIFFERENCE ALUMINUM PRICE ACF - LOLLIPOP CHART
        A simple, stem plot of ACF values out to 12 lags (~1 year). */

    var firstACFchart = am4core.create("firstACF_div", am4charts.XYChart);
    // Load adf data saved in .json format
    firstACFchart.dataSource.url = 'TimeSeries/acf.json';
    firstACFchart.dataSource.parser = new am4core.JSONParser();

    // Create X-Axis as a category axis for lollipop chart
    var categoryAxis = firstACFchart.xAxes.push(new am4charts.CategoryAxis());
    categoryAxis.dataFields.category = "Lag";
    categoryAxis.renderer.minGridDistance = 60;
    categoryAxis.title.text = 'Lag';

    // Create Y-Axis
    var valueAxis = firstACFchart.yAxes.push(new am4charts.ValueAxis());
    valueAxis.tooltip.disabled = true;
    valueAxis.renderer.minGridDistance = 40;
    valueAxis.renderer.axisFills.template.disabled = true;
    valueAxis.title.text = '[bold]ACF Value[/]';
    valueAxis.max = 1;
    valueAxis.extraMax= 0.1;

    // Highlight the area of the graph below 0.2 as red (for insignificant)
    valueAxis.events.on("datavalidated", function (ev) {
        var axis = ev.target;

        var redRange = axis.axisRanges.create();
        redRange.value = -0.20;
        redRange.endValue = 0.20;
        redRange.axisFill.fill = am4core.color("#c70700");
        redRange.axisFill.fillOpacity = 0.2;
        redRange.grid.strokeOpacity = 0;
    })

    // Fill Chart
    var series = firstACFchart.series.push(new am4charts.ColumnSeries());
    series.dataFields.categoryX = "Lag";
    series.dataFields.valueY = "ACF";

    // Set Tooltip Values and background color
    series.tooltipText = "[bold]Lag: {Lag}[/]\n[bold]ACF:[/]: {valueY.formatNumber('#.###')}";
    series.tooltip.getFillFromObject = false;
    series.tooltip.background.fill = acfColor;

    series.sequencedInterpolation = true;
    series.stroke = acfColor;
    series.fillOpacity = 0;
    series.strokeOpacity = 1;
    series.strokeDashArray = "1,3";
    series.columns.template.width = 0.02;
    series.tooltip.pointerOrientation = "horizontal";

    var bullet = series.bullets.create(am4charts.CircleBullet);
    bullet.fill = acfColor;

    firstACFchart.cursor = new am4charts.XYCursor();

 // ****************************NEW GRAPH********************************************************************//

    /* PLOT OF FIRST-DIFFERENCE ALUMINUM PRICE PACF - LOLLIPOP CHART
        A simple, stem plot of PACF values out to 12 lags (~1 year). */

    var firstPACFchart = am4core.create("firstPACF_div", am4charts.XYChart);
    // Load padf data saved in .json format
    firstPACFchart.dataSource.url = 'TimeSeries/acf.json';
    firstPACFchart.dataSource.parser = new am4core.JSONParser();

    // Create X-Axis as a category axis for lollipop chart
    var categoryAxis = firstPACFchart.xAxes.push(new am4charts.CategoryAxis());
    categoryAxis.dataFields.category = "Lag";
    categoryAxis.renderer.minGridDistance = 60;
    categoryAxis.title.text = 'Lag';

    // Create Y-Axis
    var valueAxis = firstPACFchart.yAxes.push(new am4charts.ValueAxis());
    valueAxis.tooltip.disabled = true;
    valueAxis.renderer.minGridDistance = 40;
    valueAxis.renderer.axisFills.template.disabled = true;
    valueAxis.title.text = '[bold]Partial ACF Value[/]';
    valueAxis.min = -0.25;
    valueAxis.max = 1;
    valueAxis.extraMax= 0.1;

    // Highlight the area of the graph below 0.2 as red (for insignificant)
    valueAxis.events.on("datavalidated", function (ev) {
        var axis = ev.target;

        var redRange = axis.axisRanges.create();
        redRange.value = -0.20;
        redRange.endValue = 0.20;
        redRange.axisFill.fill = am4core.color("#c70700");
        redRange.axisFill.fillOpacity = 0.2;
        redRange.grid.strokeOpacity = 0;
    })

    // Fill Chart
    var series = firstPACFchart.series.push(new am4charts.ColumnSeries());
    series.dataFields.categoryX = "Lag";
    series.dataFields.valueY = "PACF";

    // Set Tooltip Values and background color
    series.tooltipText = "[bold]Lag: {Lag}[/]\n[bold]PACF:[/]: {valueY.formatNumber('#.###')}";
    series.tooltip.getFillFromObject = false;
    series.tooltip.background.fill = pacfColor;

    series.sequencedInterpolation = true;
    series.stroke = pacfColor;
    series.fillOpacity = 0;
    series.strokeOpacity = 1;
    series.strokeDashArray = "1,3";
    series.columns.template.width = 0.02;
    series.tooltip.pointerOrientation = "horizontal";

    var bullet = series.bullets.create(am4charts.CircleBullet);
    bullet.fill = pacfColor;

    firstPACFchart.cursor = new am4charts.XYCursor();
    
// ****************************NEW GRAPH********************************************************************//

    /* PLOT OF ALUMINUM PRICE MODEL COEFFICEINT - DUMBBELL CHART
    Starting using dumbbell and clustered column demos from AMChart site, modifying both to creatre a clustered
    dumbbell chart (sans the dumbbell shape with bullet endpoints).  Added additional funcionality to show "SIG"
    classification using AMChart 'adapter' functions. */
   
    var modCOEFchart = am4core.create("modCoeff_div", am4charts.XYChart);
    // Load adf data saved in .json format
    modCOEFchart.dataSource.url = 'TimeSeries/modCoeff.json';
    modCOEFchart.dataSource.parser = new am4core.JSONParser();

    // Add legend
    modCOEFchart.legend = new am4charts.Legend();
    modCOEFchart.legend.position = 'top';
    modCOEFchart.legend.paddingBottom = 20;

    // Create axes
    var xAxis = modCOEFchart.xAxes.push(new am4charts.CategoryAxis())
    xAxis.tooltip.disabled = true;
    xAxis.dataFields.category = 'Model'
    xAxis.renderer.cellStartLocation = 0.1
    xAxis.renderer.cellEndLocation = 0.9
    xAxis.renderer.grid.template.location = 0;

    var yAxis = modCOEFchart.yAxes.push(new am4charts.ValueAxis());
    yAxis.tooltip.disabled = true;
    yAxis.title.text = 'Model Coefficients';
    yAxis.max = 2;
    yAxis.min = -2;

    // Create series data
    function createModelSeries(value, name) {
        var series = modCOEFchart.series.push(new am4charts.ColumnSeries())
        series.dataFields.categoryX = 'Model'
        // Create dumbbell with openValueY functionality
        series.dataFields.openValueY = value + "Hgh";
        series.dataFields.valueY = value + "Low"
        series.name = name
        series.sequencedInterpolation = true;
        // Need to set getFillFromObject to 'false' to color anything else
        series.tooltip.getFillFromObject = false;

        // Create labels for bars (green 'SIG' if coefficient is not 0)
        var bullet = series.bullets.push(new am4charts.LabelBullet())
        bullet.dy = 10;
        bullet.label.adapter.add("text", function(label, target, key) {
            // If the coefficient is null (not used in the model) show nothing
            if (target.dataItem && (target.dataItem.values.valueY.value == 0)) {
                return '';}
            return (target.dataItem && (target.dataItem.values.valueY.value < 0)) ? 'NOT': '[bold]SIG[/]';
        })
        bullet.label.adapter.add("fill", function(label, target, key) {
            return (target.dataItem && (target.dataItem.values.valueY.value < 0)) ? am4core.color('#fa0710'): am4core.color('#0a8f20');
        })

        // Determine color from value insert and tie to ACF and PACF for alpha and beta respectively
        if (value == 'alpha') {
            series.fill = acfColor;
            series.stroke = acfColor;
            series.tooltip.background.fill = acfColor;
            series.columns.template.tooltipText = "[bold]ALPHA Value:[/]\n[bold]High Thres:[/] {openValueY.value}\n[bold]Low Thres:[/]: {valueY.value}";
        } else if (value == 'beta') {
            series.fill = pacfColor;
            series.stroke = pacfColor;
            series.tooltip.background.fill = pacfColor;
            series.columns.template.tooltipText = "[bold]BETA Value:[/]\n[bold]High Thres:[/] {openValueY.value}\n[bold]Low Thres:[/]: {valueY.value}";
        } else {
            series.fill = contColor;
            series.stroke = contColor;
            series.tooltip.background.fill = contColor;
            series.columns.template.tooltipText = "[bold]CONSTANT Value:[/]\n[bold]High Thres:[/] {openValueY.value}\n[bold]Low Thres:[/]: {valueY.value}";
        }

        
        series.fillOpacity = 0.8;
        series.strokeOpacity = 1;

        // Position Tooltip
        series.tooltip.pointerOrientation = "horizontal";
        series.columns.template.tooltipX = am4core.percent(50);
        series.columns.template.tooltipY = am4core.percent(50);
        // If column values are zero, don't display tooltip
        series.tooltip.label.adapter.add("text", function(text, target) {
            if (target.dataItem && target.dataItem.valueY == 0) {
              return "";}
            else {
                return text;}
          });
    
        series.events.on("hidden", arrangeColumns);
        series.events.on("shown", arrangeColumns);
    
        return series;
    }

    // Create three series for alpha, beta, and constant and show as three separate dumbbell lines for each model
    createModelSeries('alpha', 'Alpha');
    createModelSeries('const', 'Constant');
    createModelSeries('beta', 'Beta');

    //modCOEFchart.cursor = new am4charts.XYCursor();

    // Define function for arranging columns
    function arrangeColumns() {

        var series = modCOEFchart.series.getIndex(0);
    
        var w = 1 - xAxis.renderer.cellStartLocation - (1 - xAxis.renderer.cellEndLocation);
        if (series.dataItems.length > 1) {
            var x0 = xAxis.getX(series.dataItems.getIndex(0), "categoryX");
            var x1 = xAxis.getX(series.dataItems.getIndex(1), "categoryX");
            var delta = ((x1 - x0) / modCOEFchart.series.length) * w;
            if (am4core.isNumber(delta)) {
                var middle = modCOEFchart.series.length / 2;
    
                var newIndex = 0;
                modCOEFchart.series.each(function(series) {
                    if (!series.isHidden && !series.isHiding) {
                        series.dummyData = newIndex;
                        newIndex++;
                    }
                    else {
                        series.dummyData = modCOEFchart.series.indexOf(series);
                    }
                })
                var visibleCount = newIndex;
                var newMiddle = visibleCount / 2;
    
                modCOEFchart.series.each(function(series) {
                    var trueIndex = modCOEFchart.series.indexOf(series);
                    var newIndex = series.dummyData;
    
                    var dx = (newIndex - trueIndex + middle - newMiddle) * delta
    
                    series.animate({ property: "dx", to: dx }, series.interpolationDuration, series.interpolationEasing);
                    series.bulletsContainer.animate({ property: "dx", to: dx }, series.interpolationDuration, series.interpolationEasing);
                })
            }
        }
    };

    // ****************************NEW GRAPH********************************************************************//

    /* PLOT OF ALUMINUM PRICE MODEL AIC - COLUMN CHART
    A simple, bare-bones column chart with 3D elements vice 2D. */

    var modAICchart = am4core.create("modAIC_div", am4charts.XYChart);
    // Load AIC data saved in .json format
    modAICchart.dataSource.url = 'TimeSeries/modCoeff.json';
    modAICchart.dataSource.parser = new am4core.JSONParser();

    // Create axes
    var xAxis = modAICchart.xAxes.push(new am4charts.CategoryAxis())
    xAxis.tooltip.disabled = true;
    xAxis.dataFields.category = 'Model'
    xAxis.renderer.minGridDistance = 20;

    var yAxis = modAICchart.yAxes.push(new am4charts.ValueAxis());
    yAxis.title.text = 'Model AIC';

    // Create series
    var series = modAICchart.series.push(new am4charts.ColumnSeries());
    series.dataFields.valueY = "AIC";
    series.dataFields.categoryX = "Model";
    series.name = "Model AIC";
    series.columns.template.tooltipText = "[bold]ARIMA{categoryX} AIC:[/]\n{valueY.formatNumber('#')}";
    series.columns.template.fillOpacity = .8;

    modAICchart.cursor = new am4charts.XYCursor();

    // ****************************NEW GRAPH********************************************************************//

    /* PLOT OF ALUMINUM PRICE MODEL FORECAST - LINE CHART
    XYChart with zoomable axis showing dates and different coloring for H-Step-Ahead forecast values (last ~30 values for model prices). */

    var forecast_chart = am4core.create("forecast_div", am4charts.XYChart);
    // Load forecast data saved in .json format (unfortuneately due to dateAxis, can only be loaded locally)
    forecast_chart.data = [
        {
            "Date": new Date(1990,0, 1),
            "Real Price": "1528",
            "Model Price": "1528"
        },
        {
            "Date": new Date(1990,1, 1),
            "Real Price": "1448.320312",
            "Model Price": "1526.48479"
        },
        {
            "Date": new Date(1990,2, 1),
            "Real Price": "1553.596423",
            "Model Price": "1505.985696"
        },
        {
            "Date": new Date(1990,3, 1),
            "Real Price": "1509.425912",
            "Model Price": "1536.932444"
        },
        {
            "Date": new Date(1990,4, 1),
            "Real Price": "1508.075136",
            "Model Price": "1515.953362"
        },
        {
            "Date": new Date(1990,5, 1),
            "Real Price": "1537.066975",
            "Model Price": "1519.525447"
        },
        {
            "Date": new Date(1990,6, 1),
            "Real Price": "1534.885057",
            "Model Price": "1524.598685"
        },
        {
            "Date": new Date(1990,7, 1),
            "Real Price": "1726.481763",
            "Model Price": "1521.203037"
        },
        {
            "Date": new Date(1990,8, 1),
            "Real Price": "1989",
            "Model Price": "1570.227207"
        },
        {
            "Date": new Date(1990,9, 1),
            "Real Price": "1859.932534",
            "Model Price": "1624.046772"
        },
        {
            "Date": new Date(1990, 10, 1),
            "Real Price": "1542.969334",
            "Model Price": "1575.129717"
        },
        {
            "Date": new Date(1990, 11, 1),
            "Real Price": "1446.013413",
            "Model Price": "1504.140571"
        },
        {
            "Date": new Date(1991,0, 1),
            "Real Price": "1434.020045",
            "Model Price": "1495.895124"
        },
        {
            "Date": new Date(1991,1, 1),
            "Real Price": "1423.497774",
            "Model Price": "1493.408503"
        },
        {
            "Date": new Date(1991,2, 1),
            "Real Price": "1414.985163",
            "Model Price": "1489.810561"
        },
        {
            "Date": new Date(1991,3, 1),
            "Real Price": "1313.69356",
            "Model Price": "1487.021535"
        },
        {
            "Date": new Date(1991,4, 1),
            "Real Price": "1222.298074",
            "Model Price": "1459.975789"
        },
        {
            "Date": new Date(1991,5, 1),
            "Real Price": "1203.328171",
            "Model Price": "1441.781996"
        },
        {
            "Date": new Date(1991,6, 1),
            "Real Price": "1214.663696",
            "Model Price": "1440.065629"
        },
        {
            "Date": new Date(1991,7, 1),
            "Real Price": "1175.775281",
            "Model Price": "1441.933293"
        },
        {
            "Date": new Date(1991,8, 1),
            "Real Price": "1128.044817",
            "Model Price": "1429.854658"
        },
        {
            "Date": new Date(1991,9, 1),
            "Real Price": "1071.441036",
            "Model Price": "1419.098976"
        },
        {
            "Date": new Date(1991, 10, 1),
            "Real Price": "1050.093931",
            "Model Price": "1405.700558"
        },
        {
            "Date": new Date(1991, 11, 1),
            "Real Price": "1014.396612",
            "Model Price": "1402.125156"
        },
        {
            "Date": new Date(1992,0, 1),
            "Real Price": "1088.983949",
            "Model Price": "1392.284379"
        },
        {
            "Date": new Date(1992,1, 1),
            "Real Price": "1168.979978",
            "Model Price": "1412.651794"
        },
        {
            "Date": new Date(1992,2, 1),
            "Real Price": "1175.110675",
            "Model Price": "1426.591514"
        },
        {
            "Date": new Date(1992,3, 1),
            "Real Price": "1204.6875",
            "Model Price": "1423.052311"
        },
        {
            "Date": new Date(1992,4, 1),
            "Real Price": "1192.595658",
            "Model Price": "1430.120327"
        },
        {
            "Date": new Date(1992,5, 1),
            "Real Price": "1160.580085",
            "Model Price": "1423.639141"
        },
        {
            "Date": new Date(1992,6, 1),
            "Real Price": "1192.735639",
            "Model Price": "1415.505763"
        },
        {
            "Date": new Date(1992,7, 1),
            "Real Price": "1182.227494",
            "Model Price": "1424.432899"
        },
        {
            "Date": new Date(1992,8, 1),
            "Real Price": "1147.784056",
            "Model Price": "1417.880327"
        },
        {
            "Date": new Date(1992,9, 1),
            "Real Price": "1062.506956",
            "Model Price": "1409.136182"
        },
        {
            "Date": new Date(1992, 10, 1),
            "Real Price": "1042.803169",
            "Model Price": "1387.784665"
        },
        {
            "Date": new Date(1992, 11, 1),
            "Real Price": "1083.278531",
            "Model Price": "1386.696524"
        },
        {
            "Date": new Date(1993,0, 1),
            "Real Price": "1078.660692",
            "Model Price": "1395.954011"
        },
        {
            "Date": new Date(1993,1, 1),
            "Real Price": "1071.989015",
            "Model Price": "1390.842504"
        },
        {
            "Date": new Date(1993,2, 1),
            "Real Price": "1023.628609",
            "Model Price": "1388.922919"
        },
        {
            "Date": new Date(1993,3, 1),
            "Real Price": "984.4232657",
            "Model Price": "1375.370876"
        },
        {
            "Date": new Date(1993,4, 1),
            "Real Price": "995.0844668",
            "Model Price": "1367.206677"
        },
        {
            "Date": new Date(1993,5, 1),
            "Real Price": "1032.096987",
            "Model Price": "1370.570756"
        },
        {
            "Date": new Date(1993,6, 1),
            "Real Price": "1061.630967",
            "Model Price": "1377.776767"
        },
        {
            "Date": new Date(1993,7, 1),
            "Real Price": "1032.646065",
            "Model Price": "1382.048664"
        },
        {
            "Date": new Date(1993,8, 1),
            "Real Price": "982.1096886",
            "Model Price": "1371.913741"
        },
        {
            "Date": new Date(1993,9, 1),
            "Real Price": "953.9982492",
            "Model Price": "1359.927021"
        },
        {
            "Date": new Date(1993, 10, 1),
            "Real Price": "908.2390061",
            "Model Price": "1354.232502"
        },
        {
            "Date": new Date(1993, 11, 1),
            "Real Price": "951.1081469",
            "Model Price": "1342.333057"
        },
        {
            "Date": new Date(1994,0, 1),
            "Real Price": "1019.651401",
            "Model Price": "1355.013123"
        },
        {
            "Date": new Date(1994,1, 1),
            "Real Price": "1104.239541",
            "Model Price": "1367.976896"
        },
        {
            "Date": new Date(1994,2, 1),
            "Real Price": "1116.11964",
            "Model Price": "1385.025753"
        },
        {
            "Date": new Date(1994,3, 1),
            "Real Price": "1106.347841",
            "Model Price": "1382.170871"
        },
        {
            "Date": new Date(1994,4, 1),
            "Real Price": "1141.922508",
            "Model Price": "1378.862889"
        },
        {
            "Date": new Date(1994,5, 1),
            "Real Price": "1208.934179",
            "Model Price": "1387.425536"
        },
        {
            "Date": new Date(1994,6, 1),
            "Real Price": "1281.832413",
            "Model Price": "1401.059522"
        },
        {
            "Date": new Date(1994,7, 1),
            "Real Price": "1246.820136",
            "Model Price": "1414.904803"
        },
        {
            "Date": new Date(1994,8, 1),
            "Real Price": "1341.688514",
            "Model Price": "1400.726375"
        },
        {
            "Date": new Date(1994,9, 1),
            "Real Price": "1445.906294",
            "Model Price": "1427.474622"
        },
        {
            "Date": new Date(1994, 10, 1),
            "Real Price": "1605.143591",
            "Model Price": "1446.038467"
        },
        {
            "Date": new Date(1994, 11, 1),
            "Real Price": "1595.555047",
            "Model Price": "1480.983913"
        },
        {
            "Date": new Date(1995,0, 1),
            "Real Price": "1744.638141",
            "Model Price": "1467.926095"
        },
        {
            "Date": new Date(1995,1, 1),
            "Real Price": "1609.972623",
            "Model Price": "1508.435616"
        },
        {
            "Date": new Date(1995,2, 1),
            "Real Price": "1517.83882",
            "Model Price": "1461.517405"
        },
        {
            "Date": new Date(1995,3, 1),
            "Real Price": "1553.387144",
            "Model Price": "1448.282925"
        },
        {
            "Date": new Date(1995,4, 1),
            "Real Price": "1479.935668",
            "Model Price": "1459.411558"
        },
        {
            "Date": new Date(1995,5, 1),
            "Real Price": "1485.655609",
            "Model Price": "1435.974327"
        },
        {
            "Date": new Date(1995,6, 1),
            "Real Price": "1560.141321",
            "Model Price": "1442.016262"
        },
        {
            "Date": new Date(1995,7, 1),
            "Real Price": "1576.067952",
            "Model Price": "1458.240753"
        },
        {
            "Date": new Date(1995,8, 1),
            "Real Price": "1467.145228",
            "Model Price": "1456.648341"
        },
        {
            "Date": new Date(1995,9, 1),
            "Real Price": "1388.75707",
            "Model Price": "1427.314566"
        },
        {
            "Date": new Date(1995, 10, 1),
            "Real Price": "1373.655791",
            "Model Price": "1413.085123"
        },
        {
            "Date": new Date(1995, 11, 1),
            "Real Price": "1373.434139",
            "Model Price": "1411.343945"
        },
        {
            "Date": new Date(1996,0, 1),
            "Real Price": "1305.886975",
            "Model Price": "1410.222575"
        },
        {
            "Date": new Date(1996,1, 1),
            "Real Price": "1311.66358",
            "Model Price": "1391.490697"
        },
        {
            "Date": new Date(1996,2, 1),
            "Real Price": "1324.079036",
            "Model Price": "1396.327754"
        },
        {
            "Date": new Date(1996,3, 1),
            "Real Price": "1296.563121",
            "Model Price": "1396.776763"
        },
        {
            "Date": new Date(1996,4, 1),
            "Real Price": "1299.340091",
            "Model Price": "1388.013422"
        },
        {
            "Date": new Date(1996,5, 1),
            "Real Price": "1208.890396",
            "Model Price": "1389.489304"
        },
        {
            "Date": new Date(1996,6, 1),
            "Real Price": "1185.686436",
            "Model Price": "1364.148228"
        },
        {
            "Date": new Date(1996,7, 1),
            "Real Price": "1189.664028",
            "Model Price": "1363.186929"
        },
        {
            "Date": new Date(1996,8, 1),
            "Real Price": "1136.554536",
            "Model Price": "1362.951813"
        },
        {
            "Date": new Date(1996,9, 1),
            "Real Price": "1076.755818",
            "Model Price": "1347.732279"
        },
        {
            "Date": new Date(1996, 10, 1),
            "Real Price": "1165.775488",
            "Model Price": "1334.662748"
        },
        {
            "Date": new Date(1996, 11, 1),
            "Real Price": "1202.856474",
            "Model Price": "1359.607678"
        },
        {
            "Date": new Date(1997,0, 1),
            "Real Price": "1259.911449",
            "Model Price": "1361.237969"
        },
        {
            "Date": new Date(1997,1, 1),
            "Real Price": "1263.066717",
            "Model Price": "1374.088087"
        },
        {
            "Date": new Date(1997,2, 1),
            "Real Price": "1301.476759",
            "Model Price": "1370.060103"
        },
        {
            "Date": new Date(1997,3, 1),
            "Real Price": "1244.790432",
            "Model Price": "1379.544255"
        },
        {
            "Date": new Date(1997,4, 1),
            "Real Price": "1296.090486",
            "Model Price": "1360.878552"
        },
        {
            "Date": new Date(1997,5, 1),
            "Real Price": "1247.295754",
            "Model Price": "1377.49752"
        },
        {
            "Date": new Date(1997,6, 1),
            "Real Price": "1265.13299",
            "Model Price": "1359.027963"
        },
        {
            "Date": new Date(1997,7, 1),
            "Real Price": "1355.896708",
            "Model Price": "1366.922983"
        },
        {
            "Date": new Date(1997,8, 1),
            "Real Price": "1274.698665",
            "Model Price": "1386.886218"
        },
        {
            "Date": new Date(1997,9, 1),
            "Real Price": "1268.941618",
            "Model Price": "1359.151373"
        },
        {
            "Date": new Date(1997, 10, 1),
            "Real Price": "1260.274776",
            "Model Price": "1363.332511"
        },
        {
            "Date": new Date(1997, 11, 1),
            "Real Price": "1206.067892",
            "Model Price": "1358.487278"
        },
        {
            "Date": new Date(1998,0, 1),
            "Real Price": "1169.123823",
            "Model Price": "1344.178195"
        },
        {
            "Date": new Date(1998,1, 1),
            "Real Price": "1153.147029",
            "Model Price": "1336.796303"
        },
        {
            "Date": new Date(1998,2, 1),
            "Real Price": "1132.009724",
            "Model Price": "1333.05341"
        },
        {
            "Date": new Date(1998,3, 1),
            "Real Price": "1114.288725",
            "Model Price": "1327.029803"
        },
        {
            "Date": new Date(1998,4, 1),
            "Real Price": "1069.804822",
            "Model Price": "1322.482787"
        },
        {
            "Date": new Date(1998,5, 1),
            "Real Price": "1022.446468",
            "Model Price": "1310.616476"
        },
        {
            "Date": new Date(1998,6, 1),
            "Real Price": "1025.520802",
            "Model Price": "1299.902208"
        },
        {
            "Date": new Date(1998,7, 1),
            "Real Price": "1022.614022",
            "Model Price": "1301.960817"
        },
        {
            "Date": new Date(1998,8, 1),
            "Real Price": "1046.425137",
            "Model Price": "1299.158646"
        },
        {
            "Date": new Date(1998,9, 1),
            "Real Price": "1014.16922",
            "Model Price": "1304.541239"
        },
        {
            "Date": new Date(1998, 10, 1),
            "Real Price": "1006.403146",
            "Model Price": "1293.270636"
        },
        {
            "Date": new Date(1998, 11, 1),
            "Real Price": "969.7404736",
            "Model Price": "1292.663752"
        },
        {
            "Date": new Date(1999,0, 1),
            "Real Price": "943.9799825",
            "Model Price": "1281.803369"
        },
        {
            "Date": new Date(1999,1, 1),
            "Real Price": "918.4915756",
            "Model Price": "1276.426253"
        },
        {
            "Date": new Date(1999,2, 1),
            "Real Price": "912.8192575",
            "Model Price": "1269.698468"
        },
        {
            "Date": new Date(1999,3, 1),
            "Real Price": "982.0727848",
            "Model Price": "1268.456821"
        },
        {
            "Date": new Date(1999,4, 1),
            "Real Price": "1016.260701",
            "Model Price": "1285.213006"
        },
        {
            "Date": new Date(1999,5, 1),
            "Real Price": "1010.416553",
            "Model Price": "1288.215865"
        },
        {
            "Date": new Date(1999,6, 1),
            "Real Price": "1073.35851",
            "Model Price": "1284.40763"
        },
        {
            "Date": new Date(1999,7, 1),
            "Real Price": "1084.772805",
            "Model Price": "1300.193167"
        },
        {
            "Date": new Date(1999,8, 1),
            "Real Price": "1133.722844",
            "Model Price": "1297.54499"
        },
        {
            "Date": new Date(1999,9, 1),
            "Real Price": "1117.48959",
            "Model Price": "1309.403339"
        },
        {
            "Date": new Date(1999, 10, 1),
            "Real Price": "1114.816252",
            "Model Price": "1300.607165"
        },
        {
            "Date": new Date(1999, 11, 1),
            "Real Price": "1174.340936",
            "Model Price": "1300.678912"
        },
        {
            "Date": new Date(2000,0, 1),
            "Real Price": "1265.096722",
            "Model Price": "1314.573099"
        },
        {
            "Date": new Date(2000,1, 1),
            "Real Price": "1259.56875",
            "Model Price": "1332.979376"
        },
        {
            "Date": new Date(2000,2, 1),
            "Real Price": "1177.801754",
            "Model Price": "1325.260732"
        },
        {
            "Date": new Date(2000,3, 1),
            "Real Price": "1088.297177",
            "Model Price": "1304.55319"
        },
        {
            "Date": new Date(2000,4, 1),
            "Real Price": "1093.33128",
            "Model Price": "1285.206719"
        },
        {
            "Date": new Date(2000,5, 1),
            "Real Price": "1117.566915",
            "Model Price": "1290.010624"
        },
        {
            "Date": new Date(2000,6, 1),
            "Real Price": "1154.938994",
            "Model Price": "1293.531855"
        },
        {
            "Date": new Date(2000,7, 1),
            "Real Price": "1129.807092",
            "Model Price": "1300.790327"
        },
        {
            "Date": new Date(2000,8, 1),
            "Real Price": "1176.81143",
            "Model Price": "1290.87997"
        },
        {
            "Date": new Date(2000,9, 1),
            "Real Price": "1100.683125",
            "Model Price": "1304.116279"
        },
        {
            "Date": new Date(2000, 10, 1),
            "Real Price": "1079.983953",
            "Model Price": "1279.438969"
        },
        {
            "Date": new Date(2000, 11, 1),
            "Real Price": "1145.787586",
            "Model Price": "1278.954838"
        },
        {
            "Date": new Date(2001,0, 1),
            "Real Price": "1176.163737",
            "Model Price": "1294.62052"
        },
        {
            "Date": new Date(2001,1, 1),
            "Real Price": "1164.183665",
            "Model Price": "1296.918065"
        },
        {
            "Date": new Date(2001,2, 1),
            "Real Price": "1094.843968",
            "Model Price": "1291.702289"
        },
        {
            "Date": new Date(2001,3, 1),
            "Real Price": "1083.810232",
            "Model Price": "1273.567026"
        },
        {
            "Date": new Date(2001,4, 1),
            "Real Price": "1109.164048",
            "Model Price": "1273.892436"
        },
        {
            "Date": new Date(2001,5, 1),
            "Real Price": "1055.324986",
            "Model Price": "1278.86425"
        },
        {
            "Date": new Date(2001,6, 1),
            "Real Price": "1019.585106",
            "Model Price": "1262.106056"
        },
        {
            "Date": new Date(2001,7, 1),
            "Real Price": "990.2778774",
            "Model Price": "1255.671049"
        },
        {
            "Date": new Date(2001,8, 1),
            "Real Price": "963.5842925",
            "Model Price": "1248.227668"
        },
        {
            "Date": new Date(2001,9, 1),
            "Real Price": "921.4534371",
            "Model Price": "1241.723066"
        },
        {
            "Date": new Date(2001, 10, 1),
            "Real Price": "958.5518566",
            "Model Price": "1230.974012"
        },
        {
            "Date": new Date(2001, 11, 1),
            "Real Price": "969.3475197",
            "Model Price": "1241.86022"
        },
        {
            "Date": new Date(2002,0, 1),
            "Real Price": "983.9553384",
            "Model Price": "1240.321542"
        },
        {
            "Date": new Date(2002,1, 1),
            "Real Price": "982.0938202",
            "Model Price": "1242.991285"
        },
        {
            "Date": new Date(2002,2, 1),
            "Real Price": "1003.560714",
            "Model Price": "1240.301633"
        },
        {
            "Date": new Date(2002,3, 1),
            "Real Price": "974.4795235",
            "Model Price": "1245.047473"
        },
        {
            "Date": new Date(2002,4, 1),
            "Real Price": "954.9553051",
            "Model Price": "1234.764754"
        },
        {
            "Date": new Date(2002,5, 1),
            "Real Price": "963.3017817",
            "Model Price": "1230.854272"
        },
        {
            "Date": new Date(2002,6, 1),
            "Real Price": "947.654529",
            "Model Price": "1232.5159"
        },
        {
            "Date": new Date(2002,7, 1),
            "Real Price": "913.5902256",
            "Model Price": "1226.514462"
        },
        {
            "Date": new Date(2002,8, 1),
            "Real Price": "917.9576881",
            "Model Price": "1217.725737"
        },
        {
            "Date": new Date(2002,9, 1),
            "Real Price": "922.7994529",
            "Model Price": "1219.620432"
        },
        {
            "Date": new Date(2002, 10, 1),
            "Real Price": "964.6914601",
            "Model Price": "1218.869063"
        },
        {
            "Date": new Date(2002, 11, 1),
            "Real Price": "964.9196527",
            "Model Price": "1228.406439"
        },
        {
            "Date": new Date(2003,0, 1),
            "Real Price": "963.0706089",
            "Model Price": "1224.478414"
        },
        {
            "Date": new Date(2003,1, 1),
            "Real Price": "987.1770833",
            "Model Price": "1223.502048"
        },
        {
            "Date": new Date(2003,2, 1),
            "Real Price": "961.2930164",
            "Model Price": "1228.487971"
        },
        {
            "Date": new Date(2003,3, 1),
            "Real Price": "928.6134007",
            "Model Price": "1218.971677"
        },
        {
            "Date": new Date(2003,4, 1),
            "Real Price": "976.2202365",
            "Model Price": "1211.452847"
        },
        {
            "Date": new Date(2003,5, 1),
            "Real Price": "982.2003004",
            "Model Price": "1224.225463"
        },
        {
            "Date": new Date(2003,6, 1),
            "Real Price": "1000.066714",
            "Model Price": "1220.949715"
        },
        {
            "Date": new Date(2003,7, 1),
            "Real Price": "1007.025407",
            "Model Price": "1224.914268"
        },
        {
            "Date": new Date(2003,8, 1),
            "Real Price": "975.7623029",
            "Model Price": "1224.175098"
        },
        {
            "Date": new Date(2003,9, 1),
            "Real Price": "1018.637366",
            "Model Price": "1214.748497"
        },
        {
            "Date": new Date(2003, 10, 1),
            "Real Price": "1041.768041",
            "Model Price": "1226.789171"
        },
        {
            "Date": new Date(2003, 11, 1),
            "Real Price": "1070.710339",
            "Model Price": "1228.148339"
        },
        {
            "Date": new Date(2004,0, 1),
            "Real Price": "1101.094145",
            "Model Price": "1233.782303"
        },
        {
            "Date": new Date(2004,1, 1),
            "Real Price": "1150.86335",
            "Model Price": "1238.681917"
        },
        {
            "Date": new Date(2004,2, 1),
            "Real Price": "1129.410365",
            "Model Price": "1248.796298"
        },
        {
            "Date": new Date(2004,3, 1),
            "Real Price": "1178.170758",
            "Model Price": "1239.099253"
        },
        {
            "Date": new Date(2004,4, 1),
            "Real Price": "1101.059378",
            "Model Price": "1252.73542"
        },
        {
            "Date": new Date(2004,5, 1),
            "Real Price": "1135.303161",
            "Model Price": "1227.699673"
        },
        {
            "Date": new Date(2004,6, 1),
            "Real Price": "1151.505637",
            "Model Price": "1241.548915"
        },
        {
            "Date": new Date(2004,7, 1),
            "Real Price": "1140.270122",
            "Model Price": "1240.643632"
        },
        {
            "Date": new Date(2004,8, 1),
            "Real Price": "1162.816122",
            "Model Price": "1236.450965"
        },
        {
            "Date": new Date(2004,9, 1),
            "Real Price": "1223.116015",
            "Model Price": "1241.866056"
        },
        {
            "Date": new Date(2004, 10, 1),
            "Real Price": "1208.716567",
            "Model Price": "1254.576244"
        },
        {
            "Date": new Date(2004, 11, 1),
            "Real Price": "1232.373314",
            "Model Price": "1246.034583"
        },
        {
            "Date": new Date(2005,0, 1),
            "Real Price": "1221.880545",
            "Model Price": "1252.864765"
        },
        {
            "Date": new Date(2005,1, 1),
            "Real Price": "1247.680938",
            "Model Price": "1246.85968"
        },
        {
            "Date": new Date(2005,2, 1),
            "Real Price": "1312.3058",
            "Model Price": "1253.588019"
        },
        {
            "Date": new Date(2005,3, 1),
            "Real Price": "1245.371617",
            "Model Price": "1267.078805"
        },
        {
            "Date": new Date(2005,4, 1),
            "Real Price": "1146.874354",
            "Model Price": "1244.71853"
        },
        {
            "Date": new Date(2005,5, 1),
            "Real Price": "1140.024552",
            "Model Price": "1223.469642"
        },
        {
            "Date": new Date(2005,6, 1),
            "Real Price": "1166.577182",
            "Model Price": "1225.68648"
        },
        {
            "Date": new Date(2005,7, 1),
            "Real Price": "1216.661258",
            "Model Price": "1230.478779"
        },
        {
            "Date": new Date(2005,8, 1),
            "Real Price": "1178.601009",
            "Model Price": "1240.702585"
        },
        {
            "Date": new Date(2005,9, 1),
            "Real Price": "1238.589725",
            "Model Price": "1226.6728"
        },
        {
            "Date": new Date(2005, 10, 1),
            "Real Price": "1323.895674",
            "Model Price": "1244.342184"
        },
        {
            "Date": new Date(2005, 11, 1),
            "Real Price": "1448.711509",
            "Model Price": "1260.357457"
        },
        {
            "Date": new Date(2006,0, 1),
            "Real Price": "1524.691689",
            "Model Price": "1287.041881"
        },
        {
            "Date": new Date(2006,1, 1),
            "Real Price": "1568.732761",
            "Model Price": "1298.303464"
        },
        {
            "Date": new Date(2006,2, 1),
            "Real Price": "1553.034443",
            "Model Price": "1305.284254"
        },
        {
            "Date": new Date(2006,3, 1),
            "Real Price": "1666.875623",
            "Model Price": "1297.890926"
        },
        {
            "Date": new Date(2006,4, 1),
            "Real Price": "1806.453587",
            "Model Price": "1327.798061"
        },
        {
            "Date": new Date(2006,5, 1),
            "Real Price": "1573.81915",
            "Model Price": "1354.708046"
        },
        {
            "Date": new Date(2006,6, 1),
            "Real Price": "1578.406851",
            "Model Price": "1285.922432"
        },
        {
            "Date": new Date(2006,7, 1),
            "Real Price": "1539.979955",
            "Model Price": "1303.424593"
        },
        {
            "Date": new Date(2006,8, 1),
            "Real Price": "1561.925895",
            "Model Price": "1287.413327"
        },
        {
            "Date": new Date(2006,9, 1),
            "Real Price": "1677.990764",
            "Model Price": "1295.736111"
        },
        {
            "Date": new Date(2006, 10, 1),
            "Real Price": "1705.556368",
            "Model Price": "1322.146192"
        },
        {
            "Date": new Date(2006, 11, 1),
            "Real Price": "1772.614767",
            "Model Price": "1320.930478"
        },
        {
            "Date": new Date(2007,0, 1),
            "Real Price": "1754.253327",
            "Model Price": "1337.110995"
        },
        {
            "Date": new Date(2007,1, 1),
            "Real Price": "1772.442662",
            "Model Price": "1326.642977"
        },
        {
            "Date": new Date(2007,2, 1),
            "Real Price": "1712.363324",
            "Model Price": "1332.555365"
        },
        {
            "Date": new Date(2007,3, 1),
            "Real Price": "1744.377042",
            "Model Price": "1313.935992"
        },
        {
            "Date": new Date(2007,4, 1),
            "Real Price": "1729.520965",
            "Model Price": "1325.544194"
        },
        {
            "Date": new Date(2007,5, 1),
            "Real Price": "1649.666388",
            "Model Price": "1317.1698"
        },
        {
            "Date": new Date(2007,6, 1),
            "Real Price": "1681.606677",
            "Model Price": "1297.127893"
        },
        {
            "Date": new Date(2007,7, 1),
            "Real Price": "1542.646592",
            "Model Price": "1309.085765"
        },
        {
            "Date": new Date(2007,8, 1),
            "Real Price": "1464.215351",
            "Model Price": "1268.454661"
        },
        {
            "Date": new Date(2007,9, 1),
            "Real Price": "1489.927375",
            "Model Price": "1257.142167"
        },
        {
            "Date": new Date(2007, 10, 1),
            "Real Price": "1516.178201",
            "Model Price": "1265.223211"
        },
        {
            "Date": new Date(2007, 11, 1),
            "Real Price": "1436.833456",
            "Model Price": "1268.417363"
        },
        {
            "Date": new Date(2008,0, 1),
            "Real Price": "1475.939264",
            "Model Price": "1245.509187"
        },
        {
            "Date": new Date(2008,1, 1),
            "Real Price": "1669.466584",
            "Model Price": "1260.067163"
        },
        {
            "Date": new Date(2008,2, 1),
            "Real Price": "1799.205008",
            "Model Price": "1304.938386"
        },
        {
            "Date": new Date(2008,3, 1),
            "Real Price": "1768.817467",
            "Model Price": "1325.419606"
        },
        {
            "Date": new Date(2008,4, 1),
            "Real Price": "1723.009281",
            "Model Price": "1310.719901"
        },
        {
            "Date": new Date(2008,5, 1),
            "Real Price": "1740.081318",
            "Model Price": "1301.141793"
        },
        {
            "Date": new Date(2008,6, 1),
            "Real Price": "1785.717512",
            "Model Price": "1306.53394"
        },
        {
            "Date": new Date(2008,7, 1),
            "Real Price": "1610.619599",
            "Model Price": "1315.449443"
        },
        {
            "Date": new Date(2008,8, 1),
            "Real Price": "1470.363881",
            "Model Price": "1266.240444"
        },
        {
            "Date": new Date(2008,9, 1),
            "Real Price": "1246.842569",
            "Model Price": "1241.127156"
        },
        {
            "Date": new Date(2008, 10, 1),
            "Real Price": "1110.864379",
            "Model Price": "1188.187269"
        },
        {
            "Date": new Date(2008, 11, 1),
            "Real Price": "907.3554386",
            "Model Price": "1165.149653"
        },
        {
            "Date": new Date(2009,0, 1),
            "Real Price": "854.4957099",
            "Model Price": "1116.858716"
        },
        {
            "Date": new Date(2009,1, 1),
            "Real Price": "802.0637444",
            "Model Price": "1114.159337"
        },
        {
            "Date": new Date(2009,2, 1),
            "Real Price": "802.8666183",
            "Model Price": "1099.754113"
        },
        {
            "Date": new Date(2009,3, 1),
            "Real Price": "858.2433924",
            "Model Price": "1102.180636"
        },
        {
            "Date": new Date(2009,4, 1),
            "Real Price": "877.8462281",
            "Model Price": "1114.389415"
        },
        {
            "Date": new Date(2009,5, 1),
            "Real Price": "937.6715735",
            "Model Price": "1114.790644"
        },
        {
            "Date": new Date(2009,6, 1),
            "Real Price": "991.9430182",
            "Model Price": "1128.677361"
        },
        {
            "Date": new Date(2009,7, 1),
            "Real Price": "1140.772732",
            "Model Price": "1137.629326"
        },
        {
            "Date": new Date(2009,8, 1),
            "Real Price": "1084.210257",
            "Model Price": "1172.368537"
        },
        {
            "Date": new Date(2009,9, 1),
            "Real Price": "1104.559689",
            "Model Price": "1147.189164"
        },
        {
            "Date": new Date(2009, 10, 1),
            "Real Price": "1148.346122",
            "Model Price": "1157.474405"
        },
        {
            "Date": new Date(2009, 11, 1),
            "Real Price": "1279.56809",
            "Model Price": "1164.642251"
        },
        {
            "Date": new Date(2010,0, 1),
            "Real Price": "1307.379459",
            "Model Price": "1195.280191"
        },
        {
            "Date": new Date(2010,1, 1),
            "Real Price": "1204.854198",
            "Model Price": "1193.032372"
        },
        {
            "Date": new Date(2010,2, 1),
            "Real Price": "1296.704579",
            "Model Price": "1165.526605"
        },
        {
            "Date": new Date(2010,3, 1),
            "Real Price": "1357.250734",
            "Model Price": "1194.946894"
        },
        {
            "Date": new Date(2010,4, 1),
            "Real Price": "1199.781822",
            "Model Price": "1201.499087"
        },
        {
            "Date": new Date(2010,5, 1),
            "Real Price": "1132.60033",
            "Model Price": "1157.471825"
        },
        {
            "Date": new Date(2010,6, 1),
            "Real Price": "1165.429542",
            "Model Price": "1149.955351"
        },
        {
            "Date": new Date(2010,7, 1),
            "Real Price": "1234.753379",
            "Model Price": "1158.897197"
        },
        {
            "Date": new Date(2010,8, 1),
            "Real Price": "1268.278574",
            "Model Price": "1173.032183"
        },
        {
            "Date": new Date(2010,9, 1),
            "Real Price": "1363.379222",
            "Model Price": "1176.542652"
        },
        {
            "Date": new Date(2010, 10, 1),
            "Real Price": "1349.404779",
            "Model Price": "1198.766378"
        },
        {
            "Date": new Date(2010, 11, 1),
            "Real Price": "1362.872853",
            "Model Price": "1187.869092"
        },
        {
            "Date": new Date(2011,0, 1),
            "Real Price": "1406.32926",
            "Model Price": "1192.669053"
        },
        {
            "Date": new Date(2011,1, 1),
            "Real Price": "1445.239029",
            "Model Price": "1201.173071"
        },
        {
            "Date": new Date(2011,2, 1),
            "Real Price": "1460.802929",
            "Model Price": "1207.53862"
        },
        {
            "Date": new Date(2011,3, 1),
            "Real Price": "1517.653943",
            "Model Price": "1208.407498"
        },
        {
            "Date": new Date(2011,4, 1),
            "Real Price": "1467.352267",
            "Model Price": "1221.402101"
        },
        {
            "Date": new Date(2011,5, 1),
            "Real Price": "1450.648888",
            "Model Price": "1203.481351"
        },
        {
            "Date": new Date(2011,6, 1),
            "Real Price": "1428.56826",
            "Model Price": "1202.281667"
        },
        {
            "Date": new Date(2011,7, 1),
            "Real Price": "1342.619446",
            "Model Price": "1195.354398"
        },
        {
            "Date": new Date(2011,8, 1),
            "Real Price": "1290.468646",
            "Model Price": "1173.357872"
        },
        {
            "Date": new Date(2011,9, 1),
            "Real Price": "1226.16349",
            "Model Price": "1164.027082"
        },
        {
            "Date": new Date(2011, 10, 1),
            "Real Price": "1167.402603",
            "Model Price": "1148.26326"
        },
        {
            "Date": new Date(2011, 11, 1),
            "Real Price": "1135.922915",
            "Model Price": "1135.603792"
        },
        {
            "Date": new Date(2012,0, 1),
            "Real Price": "1203.882515",
            "Model Price": "1129.210635"
        },
        {
            "Date": new Date(2012,1, 1),
            "Real Price": "1232.911172",
            "Model Price": "1146.966654"
        },
        {
            "Date": new Date(2012,2, 1),
            "Real Price": "1217.096872",
            "Model Price": "1148.373159"
        },
        {
            "Date": new Date(2012,3, 1),
            "Real Price": "1139.612723",
            "Model Price": "1142.394549"
        },
        {
            "Date": new Date(2012,4, 1),
            "Real Price": "1116.340775",
            "Model Price": "1122.34607"
        },
        {
            "Date": new Date(2012,5, 1),
            "Real Price": "1051.981094",
            "Model Price": "1119.99538"
        },
        {
            "Date": new Date(2012,6, 1),
            "Real Price": "1046.510674",
            "Model Price": "1102.408278"
        },
        {
            "Date": new Date(2012,7, 1),
            "Real Price": "1022.208906",
            "Model Price": "1104.033548"
        },
        {
            "Date": new Date(2012,8, 1),
            "Real Price": "1139.213038",
            "Model Price": "1095.7984"
        },
        {
            "Date": new Date(2012,9, 1),
            "Real Price": "1086.712044",
            "Model Price": "1126.743518"
        },
        {
            "Date": new Date(2012, 10, 1),
            "Real Price": "1074.49445",
            "Model Price": "1103.600204"
        },
        {
            "Date": new Date(2012, 11, 1),
            "Real Price": "1150.683989",
            "Model Price": "1104.916793"
        },
        {
            "Date": new Date(2013,0, 1),
            "Real Price": "1121.357004",
            "Model Price": "1122.807639"
        },
        {
            "Date": new Date(2013,1, 1),
            "Real Price": "1124.052265",
            "Model Price": "1109.054205"
        },
        {
            "Date": new Date(2013,2, 1),
            "Real Price": "1049.106339",
            "Model Price": "1111.802274"
        },
        {
            "Date": new Date(2013,3, 1),
            "Real Price": "1023.656629",
            "Model Price": "1090.149831"
        },
        {
            "Date": new Date(2013,4, 1),
            "Real Price": "1007.589091",
            "Model Price": "1087.650421"
        },
        {
            "Date": new Date(2013,5, 1),
            "Real Price": "995.3044",
            "Model Price": "1082.618535"
        },
        {
            "Date": new Date(2013,6, 1),
            "Real Price": "968.7638845",
            "Model Price": "1079.223496"
        },
        {
            "Date": new Date(2013,7, 1),
            "Real Price": "991.9216194",
            "Model Price": "1071.709291"
        },
        {
            "Date": new Date(2013,8, 1),
            "Real Price": "961.5590944",
            "Model Price": "1078.143833"
        },
        {
            "Date": new Date(2013,9, 1),
            "Real Price": "990.1154308",
            "Model Price": "1067.091321"
        },
        {
            "Date": new Date(2013, 10, 1),
            "Real Price": "952.009596",
            "Model Price": "1075.842188"
        },
        {
            "Date": new Date(2013, 11, 1),
            "Real Price": "945.0695299",
            "Model Price": "1062.182355"
        },
        {
            "Date": new Date(2014,0, 1),
            "Real Price": "936.0653704",
            "Model Price": "1062.408817"
        },
        {
            "Date": new Date(2014,1, 1),
            "Real Price": "917.5813638",
            "Model Price": "1058.501155"
        },
        {
            "Date": new Date(2014,2, 1),
            "Real Price": "921.2222702",
            "Model Price": "1053.207953"
        },
        {
            "Date": new Date(2014,3, 1),
            "Real Price": "976.2888107",
            "Model Price": "1054.008343"
        },
        {
            "Date": new Date(2014,4, 1),
            "Real Price": "942.3466136",
            "Model Price": "1066.558185"
        },
        {
            "Date": new Date(2014,5, 1),
            "Real Price": "988.3465001",
            "Model Price": "1052.992867"
        },
        {
            "Date": new Date(2014,6, 1),
            "Real Price": "1045.940616",
            "Model Price": "1066.916152"
        },
        {
            "Date": new Date(2014,7, 1),
            "Real Price": "1090.237487",
            "Model Price": "1076.719831"
        },
        {
            "Date": new Date(2014,8, 1),
            "Real Price": "1068.651098",
            "Model Price": "1084.14479"
        },
        {
            "Date": new Date(2014,9, 1),
            "Real Price": "1045.104301",
            "Model Price": "1075.11023"
        },
        {
            "Date": new Date(2014, 10, 1),
            "Real Price": "1105.915878",
            "Model Price": "1069.833644"
        },
        {
            "Date": new Date(2014, 11, 1),
            "Real Price": "1030.493241",
            "Model Price": "1085.447592"
        },
        {
            "Date": new Date(2015,0, 1),
            "Real Price": "985.6427497",
            "Model Price": "1060.33693"
        },
        {
            "Date": new Date(2015,1, 1),
            "Real Price": "984.8307994",
            "Model Price": "1053.705423"
        },
        {
            "Date": new Date(2015,2, 1),
            "Real Price": "958.4348139",
            "Model Price": "1053.698563"
        },
        {
            "Date": new Date(2015,3, 1),
            "Real Price": "981.9001035",
            "Model Price": "1045.343649"
        },
        {
            "Date": new Date(2015,4, 1),
            "Real Price": "970.5234699",
            "Model Price": "1052.075805"
        },
        {
            "Date": new Date(2015,5, 1),
            "Real Price": "905.4445157",
            "Model Price": "1045.867041"
        },
        {
            "Date": new Date(2015,6, 1),
            "Real Price": "878.1781174",
            "Model Price": "1029.093474"
        },
        {
            "Date": new Date(2015,7, 1),
            "Real Price": "829.2377002",
            "Model Price": "1024.858664"
        },
        {
            "Date": new Date(2015,8, 1),
            "Real Price": "853.3726169",
            "Model Price": "1011.756365"
        },
        {
            "Date": new Date(2015,9, 1),
            "Real Price": "813.3170453",
            "Model Price": "1019.892537"
        },
        {
            "Date": new Date(2015, 10, 1),
            "Real Price": "786.3150081",
            "Model Price": "1005.886679"
        },
        {
            "Date": new Date(2015, 11, 1),
            "Real Price": "802.8789565",
            "Model Price": "1001.003035"
        },
        {
            "Date": new Date(2016,0, 1),
            "Real Price": "794.6082928",
            "Model Price": "1005.046751"
        },
        {
            "Date": new Date(2016,1, 1),
            "Real Price": "822.6139012",
            "Model Price": "1000.33982"
        },
        {
            "Date": new Date(2016,2, 1),
            "Real Price": "819.9093492",
            "Model Price": "1007.303257"
        },
        {
            "Date": new Date(2016,3, 1),
            "Real Price": "838.2344986",
            "Model Price": "1003.282235"
        },
        {
            "Date": new Date(2016,4, 1),
            "Real Price": "825.2928844",
            "Model Price": "1007.55885"
        },
        {
            "Date": new Date(2016,5, 1),
            "Real Price": "845.7681616",
            "Model Price": "1001.580908"
        },
        {
            "Date": new Date(2016,6, 1),
            "Real Price": "865.0674984",
            "Model Price": "1006.922014"
        },
        {
            "Date": new Date(2016,7, 1),
            "Real Price": "868.896554",
            "Model Price": "1009.024589"
        },
        {
            "Date": new Date(2016,8, 1),
            "Real Price": "841.8182723",
            "Model Price": "1007.956859"
        },
        {
            "Date": new Date(2016,9, 1),
            "Real Price": "878.6381174",
            "Model Price": "999.7000656"
        },
        {
            "Date": new Date(2016, 10, 1),
            "Real Price": "915.1165108",
            "Model Price": "1009.868111"
        },
        {
            "Date": new Date(2016, 11, 1),
            "Real Price": "907.8851587",
            "Model Price": "1015.172187"
        },
        {
            "Date": new Date(2017,0, 1),
            "Real Price": "937.4552875",
            "Model Price": "1010.407963"
        },
        {
            "Date": new Date(2017,1, 1),
            "Real Price": "972.8284715",
            "Model Price": "1017.791753"
        },
        {
            "Date": new Date(2017,2, 1),
            "Real Price": "994.5484292",
            "Model Price": "1023.531015"
        },
        {
            "Date": new Date(2017,3, 1),
            "Real Price": "1002.791264",
            "Model Price": "1026.157785"
        },
        {
            "Date": new Date(2017,4, 1),
            "Real Price": "999.3507398",
            "Model Price": "1026.098183"
        },
        {
            "Date": new Date(2017,5, 1),
            "Real Price": "984.260011",
            "Model Price": "1023.706683"
        },
        {
            "Date": new Date(2017,6, 1),
            "Real Price": "993.2370494",
            "Model Price": "1018.900005"
        },
        {
            "Date": new Date(2017,7, 1),
            "Real Price": "1055.551269",
            "Model Price": "1020.957347"
        },
        {
            "Date": new Date(2017,8, 1),
            "Real Price": "1084.166084",
            "Model Price": "1035.059905"
        },
        {
            "Date": new Date(2017,9, 1),
            "Real Price": "1101.792372",
            "Model Price": "1037.306075"
        },
        {
            "Date": new Date(2017, 10, 1),
            "Real Price": "1081.032735",
            "Model Price": "1039.777184"
        },
        {
            "Date": new Date(2017, 11, 1),
            "Real Price": "1070.738184",
            "Model Price": "1032.240878"
        },
        {
            "Date": new Date(2018,0, 1),
            "Real Price": "1132.756089",
            "Model Price": "1030.010768"
        },
        {
            "Date": new Date(2018,1, 1),
            "Real Price": "1115.835966",
            "Model Price": "1045.147775"
        },
        {
            "Date": new Date(2018,2, 1),
            "Real Price": "1057.354237",
            "Model Price": "1035.32379"
        },
        {
            "Date": new Date(2018,3, 1),
            "Real Price": "1148.627135",
            "Model Price": "1021.197151"
        },
        {
            "Date": new Date(2018,4, 1),
            "Real Price": "1169.154179",
            "Model Price": "1047.000075"
        },
        {
            "Date": new Date(2018,5, 1),
            "Real Price": "1135.951251",
            "Model Price": "1044.117428"
        },
        {
            "Date": new Date(2018,6, 1),
            "Real Price": "1056.259031",
            "Model Price": "1034.743611"
        },
        {
            "Date": new Date(2018,7, 1),
            "Real Price": "1039.059721",
            "Model Price": "1015.002822"
        },
        {
            "Date": new Date(2018,8, 1),
            "Real Price": "1024.549509",
            "Model Price": "1013.487611"
        },
        {
            "Date": new Date(2018,9, 1),
            "Real Price": "1023.36104",
            "Model Price": "1011.972401",
        },
        {
            "Date": new Date(2018, 10, 1),
            "Real Price": "977.6055836",
            "Model Price": "1010.457191"
        },
        {
            "Date": new Date(2018, 11, 1),
            "Real Price": "969.7245124",
            "Model Price": "1008.941981"
        },
        {
            "Date": new Date(2019,0, 1),
            "Real Price": "936.2535341",
            "Model Price": "1007.426771"
        },
        {
            "Date": new Date(2019,1, 1),
            "Real Price": "938.9723889",
            "Model Price": "1005.91156"
        },
        {
            "Date": new Date(2019,2, 1),
            "Real Price": "938.7473448",
            "Model Price": "1004.39635"
        },
        {
            "Date": new Date(2019,3, 1),
            "Real Price": "921.5343815",
            "Model Price": "1002.88114"
        },
        {
            "Date": new Date(2019,4, 1),
            "Real Price": "889.3370542",
            "Model Price": "1001.36593"
        },
        {
            "Date": new Date(2019,5, 1),
            "Real Price": "876.5210063",
            "Model Price": "999.8507195"
        },
        {
            "Date": new Date(2019,6, 1),
            "Real Price": "895.2471002",
            "Model Price": "998.3355093"
        },
        {
            "Date": new Date(2019,7, 1),
            "Real Price": "866.5401021",
            "Model Price": "996.8202991"
        },
        {
            "Date": new Date(2019,8, 1),
            "Real Price": "871.5199969",
            "Model Price": "995.3050889"
        },
        {
            "Date": new Date(2019,9, 1),
            "Real Price": "854.9750241",
            "Model Price": "993.7898787"
        },
        {
            "Date": new Date(2019, 10, 1),
            "Real Price": "877.1117318",
            "Model Price": "992.2746684"
        },
        {
            "Date": new Date(2019, 11, 1),
            "Real Price": "874.7030437",
            "Model Price": "990.7594582"
        },
        {
            "Date": new Date(2020,0, 1),
            "Real Price": "873.9097477",
            "Model Price": "989.244248"
        },
        {
            "Date": new Date(2020,1, 1),
            "Real Price": "831.5771045",
            "Model Price": "987.7290378"
        },
        {
            "Date": new Date(2020,2, 1),
            "Real Price": "796.1137526",
            "Model Price": "986.2138276"
        },
        {
            "Date": new Date(2020,3, 1),
            "Real Price": "726.5708873",
            "Model Price": "984.6986174"
        },
        {
            "Date": new Date(2020,4, 1),
            "Real Price": "730.4872835",
            "Model Price": "983.1834071"
        },
        {
            "Date": new Date(2020,5, 1),
            "Real Price": "777.3288957",
            "Model Price": "981.6681969"
        },
        {
            "Date": new Date(2020,6, 1),
            "Real Price": "810.4517029",
            "Model Price": "980.1529867"
        },
        {
            "Date": new Date(2020,7, 1),
            "Real Price": "854.1672332",
            "Model Price": "978.6377765"
        },
        {
            "Date": new Date(2020,8, 1),
            "Real Price": "854.6304408",
            "Model Price": "977.1225663"
        },
        {
            "Date": new Date(2020,9, 1),
            "Real Price": "884.1135437",
            "Model Price": "975.6073561"
        },
        {
            "Date": new Date(2020, 10, 1),
            "Real Price": "945.661526",
            "Model Price": "974.0921459"
        },
        {
            "Date": new Date(2020, 11, 1),
            "Real Price": "982.0716771",
            "Model Price": "972.5769356"
        },
        {
            "Date": new Date(2021,0, 1),
            "Real Price": "974.3580135",
            "Model Price": "971.0617254"
        }
    ];    

    // Create axes
    var dateAxis = forecast_chart.xAxes.push(new am4charts.DateAxis());

    // Create value axis
    var valueAxis = forecast_chart.yAxes.push(new am4charts.ValueAxis());

    // Create series
    var realPriceSeries = forecast_chart.series.push(new am4charts.LineSeries());
    realPriceSeries.dataFields.valueY = "Real Price";
    realPriceSeries.dataFields.dateX = "Date";
    realPriceSeries.strokeWidth = 1.8;
    realPriceSeries.stroke = cpiColor;
    realPriceSeries.tooltip.background.fill = cpiColor;
    realPriceSeries.tooltipText = "Real Price:\n[bold]{valueY.value.formatNumber('$#.##')}[/]";

    var modelPriceSeries = forecast_chart.series.push(new am4charts.LineSeries());
    modelPriceSeries.dataFields.valueY = "Model Price";
    modelPriceSeries.dataFields.dateX = "Date";
    modelPriceSeries.strokeWidth = 2;
    modelPriceSeries.stroke = aluminumColor;

    // Highlight the background after 07/2018 to better showcase H-Step-Ahead region
    dateAxis.events.on("datavalidated", function (ev) {
        var axis = ev.target;

        var range = axis.axisRanges.create();
        range.date = new Date(2018, 7, 1);
        range.endDate = new Date(2021, 0, 1);
        range.axisFill.fill = am4core.color("#ff00dd");
        range.axisFill.fillOpacity = 0.2;
        range.grid.strokeOpacity = 0;
    })

    // Change tooltip text and stroke color if Date is after July 2018 (signalling H-Step Ahead Prediction)
    var bullet = modelPriceSeries.bullets.push(new am4charts.Bullet());
    bullet.tooltipText = "{dateX}\n1-Step-Forecast:\n[bold]{valueY.value.formatNumber('$#.##')}[/]";
    bullet.fill = aluminumColor;
    
    bullet.adapter.add("fill", function(fill, target) {
        if(target.dataItem && target.dataItem.values.dateX.value > new Date(2018, 6, 1)) {
            return am4core.color("#ff00dd");
        }
        return fill;})
    bullet.adapter.add("tooltipText", function(text, target) {
        if(target.dataItem && target.dataItem.values.dateX.value > new Date(2018, 6, 1)) {
            return "{dateX}\nH-Step-Forecast:\n[bold]{valueY.value.formatNumber('$#.##')}[/]";
        }
        return text;})


    // Add scrollbar
    var scrollbarX = new am4charts.XYChartScrollbar();
    scrollbarX.series.push(modelPriceSeries);
    forecast_chart.scrollbarX = scrollbarX;

    forecast_chart.cursor = new am4charts.XYCursor();

    
    








}); // End AM4Core Function