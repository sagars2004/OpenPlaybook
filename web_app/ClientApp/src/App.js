import React, { Component } from 'react';
import { Route, Routes } from 'react-router-dom';
//import AppRoutes from './AppRoutes';
import { Layout } from './components/Layout';
import { Footer } from './components/Footer';
import { Home } from './components/Home';
import { Playbook } from './components/Playbook';
import './custom.css';

export default class App extends Component {
  static displayName = App.name;

  render() {
    return (
      <div>
        <Layout>
          <Routes>
            <Route path='/' element={<Home />} />
            <Route path='/playbook' element={<Playbook />} />
          </Routes>
        </Layout>
        <Footer />
      </div>
    );
  }
}
