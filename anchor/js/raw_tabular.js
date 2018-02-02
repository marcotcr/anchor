import React from 'react';
import classNames from 'classnames';

import { Icon } from 'antd';

class RawTabular extends React.Component {
  static propTypes = {
    // each element in data must be [fname, value, weight]
    data: React.PropTypes.array.isRequired,
    // must have length 2
    labelColors: React.PropTypes.array.isRequired,
    onlyShowActive: React.PropTypes.bool,
    rowWise: React.PropTypes.bool,
    startSmall: React.PropTypes.bool
  };
  static defaultProps = {
    onlyShowActive: false,
    rowWise: false,
    startSmall: false
  };

  toggleSmall = () => {
    const { minimized } = this.state;
    this.setState({ minimized: !minimized });
  }

  constructor() {
    super();
    this.state = { minimized : false };
  }

  componentWillMount() {
    const { startSmall } = this.props;
    this.setState({ minimized: startSmall })
  }

  renderRows() {
    const { data, colors, onlyShowActive } = this.props;
    const headerAndValues = data.map(exp => {
        const [fname, value, label] = exp;
        if (onlyShowActive && label === -1) {
          return [
            (
              <td style={{display: 'none'}} key={fname}></td>
            ),
            (
              <td style={{display: 'none'}} key={fname}></td>
            )
          ];
        }
        let style;
        if (label >= 0) {
          style = { backgroundColor: labelColors[label] };
        }
        else if (label == -2) {
          style = { backgroundColor: '#5F9EA0' };
        }
        else {
          style = { color: 'black' };
        }
        return [
            (
              <td style={style} className={classNames('table_row')} key={fname}>
                {fname}
              </td>
            ),
            (
              <td style={style} className={classNames('table_row')} key={fname}>
                {value}
              </td>
            )
        ];
      }
    );
    const header = headerAndValues.map(x => x[0]);
    const values = headerAndValues.map(x => x[1]);

    return (
      <div className={classNames('lime', 'raw_tabular')}>
        <table>
          <tbody>
            <tr className={classNames('table_head')}>
              {header}
            </tr>
            <tr>
              {values}
            </tr>
          </tbody>
        </table>
      </div>
    );
  }
  renderColumns() {
    const { data, labelColors, onlyShowActive } = this.props;
    const { minimized } = this.state
    let rows = data.map(exp => {
        const [fname, value, label] = exp;
        if (onlyShowActive && label === -1) {
          return (
            <tr style={{display: 'hidden'}} key={fname}></tr>
          );
        }
        let style;
        if (label >= 0) {
          style = { backgroundColor: labelColors[label] };
        }
        else if (label == -2) {
          style = { backgroundColor: '#5F9EA0' };
        }
        else {
          style = { color: 'black' };
        }
        const fvalue = value.includes(fname) ? value : `${fname} = ${value}`;
        return (
          <tr style={style} className={classNames('table_row')} key={fname}>
            <td>{fvalue}</td>
            {/* <td>{fname}</td>
            <td>{value}</td> */}
          </tr>
        );
      }
    );
    const iconType = minimized ? 'down' : 'up';
    if (rows.length > 4) {
      if (minimized) {
        rows = rows.slice(0, 3);
      }
      rows.push((
        <tr onClick={this.toggleSmall} className={classNames('table_row', 'show_more')} key='...'>
          <td colSpan='2'><Icon type={iconType} /></td>
        </tr>
      ));
    }
    return (
      <div className={classNames('lime', 'raw_tabular')}>
        <table>
          <tbody>
            {/* <tr className={classNames('table_head')}>
              <td>Feature</td>
              <td>Value</td>
            </tr> */}
            {rows}
          </tbody>
        </table>
      </div>
    );
  }
  render() {
    const { rowWise } = this.props;
    if (rowWise) {
      return this.renderRows();
    } else {
      return this.renderColumns();
    }
  }
}

export default RawTabular;
