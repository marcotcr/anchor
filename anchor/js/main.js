import React from 'react';
import ReactDOM from 'react-dom';
import ExplanationFrame from './explanation_frame.js';
import RawTabular from './raw_tabular.js';
import RawText from './raw_text.js';
import '../css/style.less';
import 'antd/dist/antd.css';

function RenderExplanationFrame(div, labelNames, predictProba, trueClass,
                                explanation, rawData, rawDataType,
                                explanationType='anchor') {
  const labelColors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
  let showExampleFn;
  if (rawDataType === 'tabular') {
    showExampleFn = (example, other={}) => (
      <RawTabular
        data={example}
        labelColors={labelColors}
        onlyShowActive={false}
        startSmall={false}
      />
    );
  }
  else if (rawDataType === 'visualqa') {
    const imgStyle = { width: '100%' }
    showExampleFn = (example, other={}) => (
      <img src={`data:image/jpeg;base64, ${example.image}`} style={imgStyle} />
    );
  }
  else if (rawDataType === 'text') {
    showExampleFn = (example, other={}) => (
      <RawText
        data={example}
        other={other}
        colors={labelColors}
      />
    );
  }
  ReactDOM.render(
    <ExplanationFrame
      labelNames = {labelNames}
      labelColors = {labelColors}
      predictProba = {predictProba}
      trueClass = {trueClass}
      rawData = {rawData}
      rawDataType = {rawDataType}
      explanationType = {explanationType}
      explanation={explanation}
      showExampleFn={showExampleFn}
    />,
    div.node());
}

export {RenderExplanationFrame};
