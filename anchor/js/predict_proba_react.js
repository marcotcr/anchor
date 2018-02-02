import React from 'react';
import d3 from 'd3';
// import {range, sortBy} from 'lodash';
import range from 'lodash/range'
import sortBy from 'lodash/sortBy'
import classNames from 'classnames';

class PredictProbaD3 {
  constructor(div, labelNames, labelColors, title, trueClass=false) {
    let svg = d3.select(div).select('svg');
    let width = parseInt(svg.style('width'));
    let barX = width - 125;
    let bar_width = width - barX - 32;
    let barHeight = 17;
    let spaceBetweenBars = 5;
    let barYshift= title === '' ? 0 : 35;
    let numClasses = labelNames.length;
    let numBars = Math.min(5, numClasses);
    this.labelColors = labelColors;

    this.labelNamesWidth = barX;
    this.xScale = d3.scale.linear().range([0, bar_width]);
    this.rect = [];
    this.classText = [];
    this.probText = [];
    this.barX = barX;

    const trueClassHeight = trueClass === false ? 0 : 55;
    const svgHeight = numBars * (barHeight + spaceBetweenBars) + barYshift + trueClassHeight;
    svg.style('height', svgHeight + 'px');
    if (title !== '') {
      svg.append('text')
        .text(title)
        .attr('x', 20)
        .attr('y', 20);
    }
    let barY = i => (barHeight + spaceBetweenBars) * i + barYshift;
    let bar = svg.append("g");
    if (trueClass !== false) {
      let trueClassG = svg.append("g");
      const lastBar = Math.min(5, numClasses);
      const baseline = barY(lastBar);
      trueClassG.append('text')
                    .attr('x', 20)
                    .attr('y', baseline + 20)
                    .text('True Class:');
      this.trueClassCircle = trueClassG.append('circle')
                    .attr('cx', 50)
                    .attr('cy', baseline + 40)
                    .attr('r', 15)
                    // .attr('fill', this.predict_proba.colors_i()
      this.trueClassText = trueClassG.append('text')
        .attr('x', 70)
        .attr('y', baseline + 45);
    }
    for (let i of range(Math.min(5, numClasses))) {
      let rect = bar.append("rect");
      rect.attr("x", barX)
          .attr("y", barY(i))
          .attr("height", barHeight);
      this.rect.push(rect);
      bar.append("rect").attr("x", barX)
          .attr("y", barY(i))
          .attr("height", barHeight)
          .attr("width", bar_width - 1)
          .attr("fill-opacity", 0)
          .attr("stroke", "black");
      const probText = bar.append("text");
      probText.attr("y", barY(i) + barHeight - 3)
      this.probText.push(probText)

      const classText = bar.append("text");
      classText.attr("x", barX - 10)
          .attr("y", barY(i) + barHeight - 3)
          .attr("text-anchor", "end")
      this.classText.push(classText);
    }
  }
  update(labelNames, predictProba, trueClass=false) {
    if (trueClass !== false) {
      this.trueClassCircle.attr('fill', this.labelColors[trueClass]);
      this.trueClassText.text(labelNames[trueClass]);
    }
    if (labelNames.length != this.classText.length) {
      throw `Class names length ${labelNames.length} != ${this.classText.length}`;
    }
    if (predictProba.length != this.classText.length) {
      throw `Predict proba length ${predictProba.length} != ${this.classText.length}`;
    }
    for (let i of range(labelNames.length)) {
      let color = this.labelColors[i];
      if (labelNames[i] == 'Other') {
          color = '#5F9EA0';
      }
      this.rect[i].attr("width", this.xScale(predictProba[i]))
                   .style("fill", color);
      this.probText[i].attr("x", this.barX + this.xScale(predictProba[i]) + 5)
          .text(predictProba[i].toFixed(2));
      const classText = this.classText[i];
      classText.text(labelNames[i]);
      while (classText.node().getBBox()['width'] + 1 > (this.labelNamesWidth- 10)) {
        const curText = classText.text().slice(0, classText.text().length - 5);
        classText.text(curText + '...');
      }
    }
  }
}
class PredictProbaReact extends React.Component {
  static propTypes = {
    labelNames : React.PropTypes.array.isRequired,
    labelColors : React.PropTypes.array.isRequired,
    predictProba: React.PropTypes.array.isRequired,
    title: React.PropTypes.string,
    trueClass: React.PropTypes.oneOfType([React.PropTypes.number, React.PropTypes.bool]),
  };
  static defaultProps = {
    title: 'Prediction Probabilities',
    trueClass: false,
  };
  componentDidMount() {
    const { labelNames, title, predictProba, trueClass, labelColors } = this.props;
    this.predictProbaD3 = new PredictProbaD3(this.node, labelNames, labelColors, title, trueClass)
    let [ labelNamesNew, predictProbaNew ] = this.mapClasses(labelNames, predictProba);
    this.predictProbaD3.update(labelNamesNew, predictProbaNew, trueClass);
  }
  render() {
    return (
      <div className={classNames('lime', 'predict_proba')} ref={node => this.node = node}>
        <svg></svg>
      </div>
    );
  }
  componentDidUpdate() {
    let { labelNames, predictProba } = this.props;
    let [ labelNamesNew, predictProbaNew ] = this.mapClasses(labelNames, predictProba);
    this.predictProbaD3.update(labelNamesNew, predictProbaNew);
  }
  mapClasses(labelNames, predictProba) {
    // Returns a list with two lists: names and predict probas
    if (labelNames.length <= 5) {
      return [labelNames, predictProba];
    }
    let classDict = predictProba.map(
      (p, i) => ({'name': labelNames[i], 'prob': p, 'i' : i}));
    let sorted = sortBy(classDict, d =>  -d.prob);
    let otherProb = 0;
    range(4, sorted.length).map(d => otherProb += sorted[d].prob);
    let ret_probs = [];
    let ret_names = [];
    for (let d of range(4)) {
      ret_probs.push(sorted[d].prob);
      ret_names.push(sorted[d].name);
    }
    ret_names.push("Other");
    ret_probs.push(otherProb);
    return [ret_names, ret_probs];
  }
}

export default PredictProbaReact;
