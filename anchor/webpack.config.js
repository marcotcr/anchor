var path = require('path');
var webpack = require('webpack');

module.exports = {
    entry: {
             bundle: './js/main.js',
           },
    output: {
        path: __dirname,
        filename: '[name].js',
        library: 'lime'
    },
    module: {
        loaders: [
            {
                loader: 'babel-loader',
                test: path.join(__dirname, 'js'),
                query: {
                  presets: ['react', 'es2015'],
                  plugins: ['transform-class-properties', ["import", [{ "libraryName": "antd" }]]],
                },
            },
            {
              test: /\.css$/,
              loaders: ['style', 'css'],
            },
            {
              test: /\.less$/,
              test: path.join(__dirname, 'css'),
              loaders: ['style', 'css', 'less'],
            }

        ]
    },
    plugins: [
        // Avoid publishing files when compilation fails
        new webpack.DefinePlugin({
         'process.env.NODE_ENV': '"production"'
        }),

        new webpack.NoErrorsPlugin()
    ],
    stats: {
        // Nice colored output
        colors: true
    },
    // Create Sourcemaps for the bundle
    devtool: 'source-map',
};
