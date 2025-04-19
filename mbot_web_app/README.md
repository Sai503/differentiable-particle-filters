# MBot Web App

Web app for the MBot robot.

## Installing from the Latest Release (Recommended)

To install the MBot web app, get the latest release from the [releases page](https://github.com/mbot-project/mbot_web_app/releases). Then, on a terminal in your MBot, install as follows:
```bash
# Note: Replace $VERSION with your desired version
# OR do "export VERSION=vX.Y.Z", substituting the latest version number.
wget https://github.com/mbot-project/mbot_web_app/releases/download/$VERSION/mbot_web_app-$VERSION.tar.gz
tar -xvzf mbot_web_app-$VERSION.tar.gz
```
Install the dependencies:
```bash
cd mbot_web_app-$VERSION/
./install_nginx.sh
./install_python_deps.sh
```
Then deploy the app:
```bash
./deploy_app.sh --no-rebuild
```
You can now safely delete the files you downloaded and extracted.

## Installing from Source

To build from source, or to develop locally, follow these steps.

### Dependencies

The front end relies on NodeJS (to compile and run the JavaScript files), NPM (a
package manager for NodeJS applications) and React, as well as some other
packages used to build and serve the files. FThere are a few ways to install NodeJS and NPM. Using `nvm` is easy and recommended. To install it, use the [install script](https://github.com/nvm-sh/nvm#install--update-script):
```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.3/install.sh | bash
```
You will have to open a new terminal after installation for the `nvm` command to be available. Install the latest version of NodeJS (as of this writing, the latest version is 18):
```bash
nvm install 18
```
Now you should have the `node` and `npm` command installed. You can check with `node --version` and `npm --version`.

We also need MBot Bridge to build the web app. Clone [MBot Bridge](https://github.com/mbot-project/mbot_bridge/). Then do:
  ```bash
  cd mbot_bridge/mbot_js
  npm install
  npm link
  ```

### Installing from Source on a Robot

To set up the webapp on a new robot from source, use the helper scripts. First, install nginx and the Python dependencies:
```bash
./scripts/install_nginx.sh
./scripts/install_python_deps.sh
```
Then, build and install the app.
```bash
./scripts/deploy_app.sh
```
By default, this will grab the latest compatible release of the [MBot Bridge](https://github.com/mbot-project/mbot_bridge/). If you want to use a local version of the Bridge API, pass the path to your copy of the MBot Bridge repository:
```bash
./scripts/deploy_app.sh --bridge-path PATH/TO/BRIDGE
```
The webapp is accessible by typing the robot's IP into the browser.

Once you have done the setup once, you only need to rerun the `deploy_app.sh` script in order to update the webapp code with a new version.

## Development Mode

If you are developing this app and want to run it locally, follow these instructions. You will need to run both the front end and the back end.

### Front end

#### Installation

First install packages and the MBot Bridge dependency:
1. Clone [MBot Bridge](https://github.com/mbot-project/mbot_bridge/). Then do:
  ```bash
  cd mbot_bridge/mbot_js
  npm install
  npm link
  ```
2. In this repo, do:
  ```bash
  npm install
  npm link mbot-js-api
  ```

#### Running

To run the React app, in the root directory of this repository, do:
```bash
npm run dev
```
This will start a development server and display the page `index.html`.
The style file is in `css/main.css`, and the JavaScript being run is in
`src/main.jsx`.

If you go to `http://[SERVER_IP]:8000` in your browser, you should see the
webapp.

### Back end

The backend is built using Flask and Python 3. If
working on a Linux computer, you probably want to run the code in a virtual
environment (on the Raspberry Pi, you can install things directly if you want).
To make a virtual environment and then activate it, do:
```bash
python3.8 -m venv ~/envs/mbot-web-app
source ~/envs/mbot-web-app/bin/activate
```
It will probably work with versions of Python 3 other than 3.8, if you don't
have it installed, but the current code was tested with Python 3.8. You can
replace `~/envs/` with your preferred path if you would like. To get out of the
virtual environment, type `deactivate`.

The setup for the Flask + React app is based off
[this tutorial](https://blog.miguelgrinberg.com/post/how-to-create-a-react--flask-project).

#### Installation

In the virtual environment (if applicable), do:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
Note: Any other Python requirements should be added to `requirements.txt`.

You will need to install the LCM messages for the MBot in this environment. See the corresponding repo for instructions. If you have already installed it globally, you might have to delete build, activate this environment, then rebuild and reinstall. This will just additionally install the messages in your current Python virtual environment. It will not mess up your previous global version.

#### Running

To run the Flask app, do:
```bash
npm run start-api
```

Traffic on `http://[SERVER_IP]:8000` will be forwarded to `http://[SERVER_IP]:5000`,
where the Flask server is running.

**Note:** If you have the webapp installed already, you will need to stop the backend service to run it locally so that the two don't conflict. To stop the installed server, do:
```bash
sudo systemctl stop mbot-web-server.service
```

## Generating a Release

To generate a new release of the web app with version `vX.Y.Z`, do the following:
1. Create a new branch with the name `vX.Y.Z-rc`.
2. Modify the `"version"` tag in `package.json` to `vX.Y.Z`.
3. Use the `generate_release.sh` script to create the prebuilt release:
  ```bash
  ./scripts/generate_release.sh -v vX.Y.Z [-b PATH/TO/BRIDGE]
  ```
  The `-b` argument is optional, and lets you pass a path to a local version of the MBot Bridge to compile against. By default, the latest compatible release of the MBot Bridge will be downloaded from source.

  In general, it is good practice to use a released version and specify this in the release notes. You may want to use a local version to generate a candidate release with an unreleased bridge API.
4. Download generated file `mbot_web_app-vX.Y.Z.tar.gz`.
5. Create a release on GitHub and upload file `mbot_web_app-vX.Y.Z.tar.gz` to the release.
