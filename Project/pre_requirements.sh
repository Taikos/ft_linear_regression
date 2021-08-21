which -s brew
if [[ $? != 0 ]] ; then
    echo "Installing Homebrew"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    echo "Installing Python Packages"
    brew install python3
    python3 -m pip install -r requirements.txt 
else
    echo "Updating Homebrew"
    brew update
    echo "Installing Python Packages"
    brew install python3
    python3 -m pip install -r requirements.txt 
fi