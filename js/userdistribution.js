function getDistribution(bucketarray, num_userbuckets) {
    console.log(bucketarray);
    var minimum = bucketarray[0]['low'];
    var maximum = bucketarray[bucketarray.length - 1]['high'];
    console.log(minimum, maximum);
    var width = (maximum - minimum) / num_userbuckets;
    var newbuckets = [];
    var widthsofar = 0;
    var frequency = 0;
    var pointer = minimum;
    for (var i = 0; i < bucketarray.length; i++) {
        if (widthsofar == width) {
            newbuckets.push({low: pointer, high: pointer+width, frequency: frequency, size: width});
            //widthsofar = bucketarray[i].size;
            //frequency = bucketarray[i].frequency;
            widthsofar = 0;
            frequency = 0;
            pointer += width;
        }
        if (widthsofar + bucketarray[i].size < width) {
            widthsofar += bucketarray[i].size;
            frequency += bucketarray[i].frequency;
        } else if (widthsofar + bucketarray[i].size == width) {
            newbuckets.push({low: pointer, high: pointer+width, size: width, frequency: frequency});
            widthsofar = 0;
            frequency = 0;
            pointer += width;
        } else {
            while (widthsofar + bucketarray[i].size > width) {
                //how much do we need to take from this bucket?
                sizeperc = (width - widthsofar) / bucketarray[i].size;
                frequency += bucketarray[i].frequency * sizeperc;
                newbuckets.push({low: pointer, high: pointer+width, frequency: frequency, size: width});
                pointer += width;
                frequency = 0;
                widthsofar = 0;
                //frequency = 0.0
                bucketarray[i].frequency -= sizeperc * bucketarray[i].frequency;
                bucketarray[i].low = pointer;
                bucketarray[i].size = bucketarray[i].high - bucketarray[i].low;
            }
            frequency = bucketarray[i].frequency;
        }
    }
    if (newbuckets.length < num_userbuckets) {
        newbuckets.push({low: pointer, high: pointer+width, frequency: frequency, size: width});
    }
    return newbuckets;
}

function loadJSON(callback) {   

    var xobj = new XMLHttpRequest();
        xobj.overrideMimeType("application/json");
    xobj.open('GET', '../beta20control.json', true); // Replace 'my_data' with the path to your file
    xobj.onreadystatechange = function () {
          if (xobj.readyState == 4 && xobj.status == "200") {
            // Required use of an anonymous callback as .open will NOT return a value but simply returns undefined in asynchronous mode
            callback(xobj.responseText);
          }
    };
    xobj.send(null);  
};



loadJSON(function(response) {
  // Parse JSON string into object
    var actual_JSON = JSON.parse(response);
    console.log(actual_JSON);
    var bucketcopy = actual_JSON.slice();
    newbuckets = getDistribution(bucketcopy, 10);
    console.log(newbuckets);
    var data = d3.range(1000).map(d3.randomBates(10));

var formatCount = d3.format(",.0f");
var formatEdges = d3.format(".3f");

var svg = d3.select("svg"),
    margin = {top: 10, right: 30, bottom: 30, left: 30},
    width = 950 - margin.left - margin.right,
    height = 500 - margin.top - margin.bottom,
    g = svg.append("g").attr("transform", "translate(" + margin.left + "," + margin.top + ")");

var x = d3.scaleLinear()
    .domain([newbuckets[0].low - (newbuckets[0].size / 2), newbuckets[newbuckets.length - 1].high + (newbuckets[0].size / 2)])
    .range([0, width]);

// var bins = d3.histogram()
//     .domain(x.domain())
//     .thresholds(x.ticks(20))
//     (data);

// console.log(bins);

var y = d3.scaleLinear()
    .domain([0, d3.max(newbuckets, function(d) { return d.frequency; })])
    .range([height, 0]);

var tip = d3.tip()
      .attr('class', 'd3-tip')
      .direction('e')
      .offset([0, 20])
      .html(function(d) {
        return '<table id="tiptable"><tr><td>Low:</td><td>' + formatEdges(d.low) + "</td></tr><tr><td>High:</td><td>" 
            + formatEdges(d.high) + "</td></tr></table>";
    });

svg.call(tip);

var bar = g.selectAll(".bar")
  .data(newbuckets)
  .enter().append("g")
    .attr("class", "bar")
    .attr("transform", function(d) { return "translate(" + 0 + "," + y(d.frequency) + ")"; })
    .on('mouseover', tip.show)
    .on('mouseout', tip.hide);

// bar.append("rect")
//     .attr("x", 1)
//     .attr("width", x(newbuckets[0].size))
//     .attr("height", function(d) { return height - y(d.frequency); });

bar.append("rect")
            .attr('x', function(d){console.log(x(d.low)); return x(d.low);})
            .attr('height', function(d){console.log(height - y(d.frequency)); return height - y(d.frequency);})
            .attr('width', x(newbuckets[0].high) - x(newbuckets[0].low))
            .style("stroke", "black")
            //.attr('height', function(d){return y(d.frequency);})
            //.attr('fill','steelblue')

bar.append("text")
    .attr("dy", ".75em")
    .attr("y", -10)
    .attr("x", function(d){return x(d.low) + ((x(d.high) - x(d.low)) / 2);})
    .attr("text-anchor", "middle")
    .text(function(d) { return formatCount(d.frequency); });

g.append("g")
    .attr("class", "axis axis--x")
    .attr("transform", "translate(0," + height + ")")
    .call(d3.axisBottom(x));

g.append("g")
    .attr("class", "axis axis--y")
    .attr("transform", "translate(0, 0)")
    .call(d3.axisLeft(y));
});