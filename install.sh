venv_dir="venv"

# Create and activate python venv
printf "\e[1m\e[36mCreate and activate python venv\e[0m\n"
if [[ -z "${VIRTUAL_ENV}" ]]
then
    if [[ ! -d "${venv_dir}" ]]
    then
        python3 -m venv "${venv_dir}"
    fi
    if [[ -f "${venv_dir}"/bin/activate ]]
    then
        source "${venv_dir}"/bin/activate
        printf "Python venv activated\n"
    else
        printf "\e[1m\e[31mERROR: Cannot activate python venv, aborting...\e[0m\n"
        exit 1
    fi
else
    printf "\e[1m\e[33mWARNING: Python venv already activated, skipping...\e[0m\n"
fi

# Installing packages
printf "\e[1m\e[36mInstall packages\e[0m\n"
if [[ -f "requirements.txt" ]]
then
    pip install -r requirements.txt
    printf "Packages installed\n"
else
    printf "\e[1m\e[31mERROR: Cannot find requirements.txt, aborting...\e[0m\n"
    exit 1
fi