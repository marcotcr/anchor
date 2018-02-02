import React from 'react';
import classNames from 'classnames';
import ExampleRoll from './example_roll.js';
import isUndefined from 'lodash/isUndefined';
import { Collapse, Button, Row, Col } from 'antd';
const Panel = Collapse.Panel;
const ButtonGroup = Button.Group;

class AnchorExamples extends React.Component {
  static propTypes = {
    // each item must have covered, coveredTrue and coveredFalse if showExamples is true
    examples: React.PropTypes.array.isRequired,
    showExampleFn: React.PropTypes.func.isRequired,
    labelName: React.PropTypes.string.isRequired,
    labelColor: React.PropTypes.string.isRequired,
    idx: React.PropTypes.number.isRequired,
    startOpen: React.PropTypes.bool,
    instanceName: React.PropTypes.string,
    other: React.PropTypes.object
  };

  static defaultProps = {
    instanceName: 'Example',
    startOpen: false,
    other: {}
  };

  render() {
    const { examples, showExampleFn, instanceName, labelName, startOpen, idx, other} = this.props;
    if (isUndefined(examples) || idx >= examples.length || examples.length === 0) {
      return (<div />);
    }
    const examplesTrue = examples[idx]['coveredTrue'].map((x) => showExampleFn(x, other));
    const examplesFalse = examples[idx]['coveredFalse'].map((x) => showExampleFn(x, other));
    return (
      <div>
        <Row gutter={16}>
          <Col span={12}>
            <Collapse defaultActiveKey={startOpen ? 'examples' : ''}>
              <Panel key='examples' header={`${instanceName}s where the A.I. agent predicts ${labelName}`}>
                <ExampleRoll
                  examples = {examplesTrue}
                />
              </Panel>
            </Collapse>
          </Col>
          <Col span={12}>
            <Collapse defaultActiveKey={startOpen ? 'examples' : ''}>
              <Panel key='examples' header={`${instanceName}s where the A.I. agent DOES NOT predict ${labelName}`}>
                <ExampleRoll
                  examples = {examplesFalse}
                />
                {examplesFalse.length === 0 && `Could not find any ${instanceName}s`}
              </Panel>
            </Collapse>
          </Col>
        </Row>
      </div>
    );
  }
}

export default AnchorExamples;
