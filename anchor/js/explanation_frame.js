import React from 'react';
import PredictProbaReact from './predict_proba_react.js';
import AnchorExamples from './anchor_examples.js';
import Anchor from './anchor.js';
import classNames from 'classnames';
import { Card, Row, Col, Popover, Icon} from 'antd';

function makeExplanationObj(labelNames, labelColors, predictProba, explanation,
                            explanationType, onChange, ifStatement) {
  const label = predictProba.indexOf(Math.max(...predictProba));
  const labelName = labelNames[label];
  const labelColor = labelColors[label];
  const { names, certainties, supports } = explanation;
  return (
    <Anchor
      names={names}
      certainties={certainties}
      supports={supports}
      labelName={labelName}
      labelColor={labelColor}
      onChange={onChange}
      ifStatement={ifStatement}
    />
  );
}

class ExplanationFrame extends React.Component {
  static propTypes = {
    labelNames : React.PropTypes.array.isRequired,
    labelColors: React.PropTypes.array.isRequired,
    predictProba: React.PropTypes.array.isRequired,
    explanation: React.PropTypes.object.isRequired,
    rawDataType: React.PropTypes.string.isRequired,
    showExampleFn: React.PropTypes.func.isRequired,
    rawData: React.PropTypes.oneOfType([React.PropTypes.string,
                                        React.PropTypes.array,
                                        React.PropTypes.object]).isRequired,
    explanationType: React.PropTypes.string,
    trueClass: React.PropTypes.oneOfType([React.PropTypes.number,
                                          React.PropTypes.bool]),
    instanceName: React.PropTypes.string,
    showInfos: React.PropTypes.bool,
    showPredictProba: React.PropTypes.bool
  };
  static defaultProps = {
    trueClass: false,
    explanationType: 'anchor',
    instanceName: 'Example',
    showPredictProba: false,
    showInfos: false
  };
  constructor() {
    super();
    this.state = {
      anchorIndex : 0,
      infoExample: true,
      infoExplanation: true,
      infoExamples: true
    };
  }
  updateAnchorIndex = (anchorIndex) => {
    this.setState({ anchorIndex });
  }

  render() {
    const { labelNames, labelColors, predictProba, trueClass, explanation, explanationType,
            rawData, rawDataType, instanceName, showPredictProba, showInfos, showExampleFn } = this.props;
    const { anchorIndex, infoExample, infoExplanation, infoExamples } = this.state;
    let explanationObj, expCoverageObj, exampleObj;
    const label = predictProba.indexOf(Math.max(...predictProba));
    const labelName = labelNames[label];
    const labelColor = labelColors[label];
    const other = {}
    let ifStatement;
    // let showExampleFn;
    if (rawDataType == 'tabular') {
      ifStatement = 'If ALL of these are true:';
    }
    else if (rawDataType == 'text') {
      other['idx'] = anchorIndex
      ifStatement = 'If ALL of these words are in the text:';
    }
    // Infos
    const contentInfoExample = (
      <div>
        This is a {instanceName}. <br />
        The A.I. makes a prediction for this {instanceName} on the right.
        {explanationType === 'none' ? (<br />) : ''}
        {explanationType === 'none' ? `\nYou can see more ${instanceName}s and predictions by clicking on the numbers above.` : ''}
        <Icon type="close" style={{marginLeft: 5}} onClick={() => this.setState({infoExample: false})}/>
      </div>
    );
    const contentInfoExamples = (
      <div>
        Here are some examples where the rule holds.
        On the left are examples where the A.I. makes the same prediction <br />
        while on the right there examples where it does not (these are rare). <br />
        I highlighted what appears in the rule.
        <Icon type="close" style={{marginLeft: 5}} onClick={() => this.setState({infoExamples : false})}/>
      </div>
    );

    const rawDataObj = showExampleFn(rawData, other);

    explanationObj = makeExplanationObj(
      labelNames, labelColors, predictProba, explanation, explanationType,
      this.updateAnchorIndex, ifStatement);
    if (explanationObj !== '') {
      explanationObj = (
        <Col span={11}>
          <Card title={'Explanation of A.I. prediction'}>
            {explanationObj}
          </Card>
        </Col>
      );
    }
    if (rawDataType === 'tabular' || rawDataType === 'text') {
      const { examples } = explanation;
      exampleObj = (
        <Popover arrowPointAtCenter placement="topLeft"
          content={contentInfoExamples}
          visible={infoExamples && showInfos} trigger="click">
          <Row style={{marginTop: 3}}>
            <Col span="24">
              <AnchorExamples
                examples={examples}
                showExampleFn={showExampleFn}
                labelName={labelName}
                labelColor={labelColor}
                instanceName={instanceName}
                other={other}
                idx={anchorIndex}
              />
            </Col>
          </Row>
        </Popover>
      );
    }
    let predictionObj;
    if (showPredictProba) {
      predictionObj = (
        <PredictProbaReact
          labelNames = {labelNames}
          predictProba = {predictProba}
          trueClass={trueClass}
          labelColors={labelColors}
        />
      )
    }
    else {
      const circleStyle = {
        backgroundColor: labelColor,
        width: 25,
        height: 25,
        marginRight: 10,
        textAlign: 'center',
        borderRadius: '50%'
      };
      const outerStyle =  {
        display: 'flex',
        alignItems: 'center'
      }
      predictionObj = (
        <div style={outerStyle}>
          <div style={circleStyle} />
          {labelName}
        </div>
      );
    }
    return (
      <div className={classNames('lime', 'explanation_frame')}>
        <Row gutter={16}>
          <Col span={8}>
            <Popover arrowPointAtCenter placement="top" content={contentInfoExample} visible={infoExample && showInfos && explanationType === 'none'} trigger="click">
              <Card title={instanceName}>
                {rawDataObj}
              </Card>
            </Popover>
          </Col>
          <Col span={5}>
            <Card title={'A.I. prediction'}>
              {predictionObj}
            </Card>
          </Col>
          {explanationObj}
        </Row>
        {exampleObj}
      </div>
    );
  }
}

export { makeExplanationObj }
export default ExplanationFrame;
