import React from 'react';
// import {range, sortBy} from 'lodash';
import range from 'lodash/range'
import sortBy from 'lodash/sortBy'
import classNames from 'classnames';

class RawText extends React.Component {
  static propTypes = {
    data: React.PropTypes.object.isRequired,
    // must have length 2
    colors: React.PropTypes.array.isRequired,
    // rawIndexes: React.PropTypes.array.isRequired,
    title: React.PropTypes.string,
    other: React.PropTypes.object
  };
  static defaultProps = {
    title: 'Text with highlighted words',
    other: {}
  };
  render() {
    const { data, colors, title, other } = this.props;
    const { text } = data
    let { rawIndexes } = data
    const { idx } = other
    rawIndexes = rawIndexes.slice(0, idx + 1)
    let wordLists = [[], []];
    for (let [word, start, weight] of rawIndexes) {
      if (weight > 0) {
        wordLists[1].push([start, start + word.length]);
      }
      else {
        wordLists[0].push([start, start + word.length]);
      }
    }
    let objects = [];
    for (let i of range(wordLists.length)) {
      wordLists[i].map(x => objects.push({'label' : i, 'start': x[0], 'end': x[1]}));
    }
    objects = sortBy(objects, x=>x['start']);
    // let node = text_span.node().childNodes[0];
    const spanize = (text, key, filled, color) => (
      <span style={{backgroundColor: filled ? color : 'transparent'}} key={key}>
      {text}
      </span>
    );
    let textList = [];
    let previous = 0;
    for (let obj of objects) {
      if (previous !== obj.start) {
        textList.push(spanize(text.slice(previous, obj.start), previous, false, false));
      }
      textList.push(spanize(text.slice(obj.start, obj.end), obj.start, true, colors[obj.label]));
      previous = obj.end;
    }
    textList.push(spanize(text.slice(previous), previous, false, false));
    const title_obj = title === '' ? '' : (<h3>{title}</h3>);
    return (
      <div className={classNames('lime', 'raw_text')}>
        {title_obj}
        {textList}
      </div>
    );
  }
}

export default RawText;
