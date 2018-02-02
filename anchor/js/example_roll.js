import React from 'react';
import classNames from 'classnames';
import { Pagination } from 'antd';

class ExampleRoll extends React.Component {
  static propTypes = {
  examples: React.PropTypes.array.isRequired,
    windowSize: React.PropTypes.number,
    onUpdateIdx: React.PropTypes.func,
  };
  static defaultProps = {
    windowSize: 1,
    onUpdateIdx: (x) => {},
  };
  constructor() {
    super();
    this.state = { exampleIdx: 0};
  }
  handlePagination = (pageNumber) => {
    const { onUpdateIdx } = this.props;
    const selected = pageNumber - 1;
    onUpdateIdx(selected);
    this.setState({exampleIdx: selected})
  }

  // componentWillReceiveProps(nextProps){
  //   if (this.props.examples !== nextProps.examples) {
  //      this.setState({exampleIdx: 0});
  //   }
  // }

  render() {
    const { examples, windowSize} = this.props;
    const { exampleIdx } = this.state
    const size = examples.length;
    if (!size) {
      return (
        <div style={{margin: 20}}/>
      );
    }
    const exSlice = examples.slice(exampleIdx, exampleIdx + windowSize).map((x, i) => (
      <div className={classNames('example_div')} key={i}>
        {x}
      </div>
    ));
    return (
      <div className={classNames('lime', 'example_roll')}>
        <div className={classNames('pagination')}>
          <Pagination
            total={size}
            defaultPageSize={1}
            current={exampleIdx + 1}
            onChange={this.handlePagination}
            size='small'
          />
        </div>
        <div className={classNames('examples_div')}>
          {exSlice}
        </div>
      </div>
    );
  }
}

export default ExampleRoll;
