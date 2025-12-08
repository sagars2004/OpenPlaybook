import React, { Component } from 'react';

export class Playbook extends Component {
  static displayName = Playbook.name;

  render() {
    return (
      <div>
        <h1>Choose Your Analysis Type</h1>
        <br></br>
        <p>Select one of the following options to get started:</p>
        <div className="text-center" style={{ marginTop: '2rem' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', alignItems: 'center' }}>
            <button className="btn btn-primary btn-lg" style={{ minWidth: '300px' }}>
              NLP Semantic Search
            </button>
            <button className="btn btn-primary btn-lg" style={{ minWidth: '300px' }}>
              CV Detection Analysis
            </button>
          </div>
        </div>
      </div>
    );
  }
}

