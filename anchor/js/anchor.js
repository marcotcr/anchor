import React from 'react';
import classNames from 'classnames';
import ExampleRoll from './example_roll.js';
import { Collapse } from 'antd';
const Panel = Collapse.Panel;
import { Button } from 'antd';
const ButtonGroup = Button.Group;

class Anchor extends React.Component {
  static propTypes = {
    names : React.PropTypes.array.isRequired,
    certainties: React.PropTypes.array.isRequired,
    supports: React.PropTypes.array.isRequired,
    labelName: React.PropTypes.string.isRequired,
    labelColor: React.PropTypes.string.isRequired,
    instanceName: React.PropTypes.string,
    ifStatement: React.PropTypes.string,
    // Takes as parameter an idx
    onChange: React.PropTypes.func
  };

  static defaultProps = {
    instanceName: 'Example',
    ifStatement: 'If ALL of these are true:',
    onChange: (i) => {}
  };

  constructor() {
    super();
    this.state = { idx: 0 };
  }

  handleChange = (idx) => {
    const { onChange } = this.props;
    onChange(idx);
    this.setState({idx});
  }

  resetState = (props) => {
    const { certainties } = props;
    this.handleChange(certainties.length - 1);
  }

  componentWillMount() {
    this.resetState(this.props);
  }
  componentWillReceiveProps(nextProps){
    if (JSON.stringify(this.props.names) !== JSON.stringify(nextProps.names)) {
       this.resetState(nextProps);
    }
  }
  render() {
    const { names, certainties, supports, labelName, labelColor,
            instanceName, ifStatement } = this.props;
    const { idx } = this.state;
    const btn_names = names.map( (x, i) => {
      const onClickFn = () => this.handleChange(i);
      const selected = i <= idx;
      return (<Button onClick={onClickFn} type={selected ? 'primary' : 'dashed'} icon={selected ? 'check' : null} key={i}>
        {x}
      </Button>
      )
    }
    );
    const precision = `${(certainties[idx] * 100).toFixed(1)}%`
    const coverage = `${(supports[idx] * 100).toFixed(1)}%`
    return (
      <div className={classNames('lime', 'anchor')}>
        <div>
          <span>{ifStatement}</span>
          <ButtonGroup>
            {btn_names}
          </ButtonGroup>
        </div>
        <div className={classNames('model_prediction')}>
          {'The A.I. will predict '}
          <span className={classNames('label')} style={{backgroundColor: labelColor}}>
            {labelName}
          </span>
          <span className={classNames('percentage')}>{precision}</span>
          {' of the time'} <br />
        </div>
      </div>
    );
  }
}

export default Anchor;
