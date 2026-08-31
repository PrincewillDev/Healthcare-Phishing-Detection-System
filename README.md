# Healthcare Phishing Detection System

Machine learning system for detecting phishing emails in healthcare
contexts. This repository currently covers data preparation only
(sourcing, cleaning, sampling). Feature engineering and model
training are not yet included.

This guide assumes no prior experience with terminals, virtual
environments, or git. Every command below is explained before you
run it, and every major step tells you what you should see if it
worked. If you get stuck, check the Troubleshooting section at the
end.

## Before you begin: terminal basics

Most of the setup below happens in a "terminal" (also called a
"command line" or, on Windows, "PowerShell"). It is a text-based way
to give your computer instructions, instead of clicking icons.

**Opening a terminal:**

- **Windows**: Click the Start menu, type `PowerShell`, and open
  "Windows PowerShell".
- **Mac**: Press `Cmd + Space` to open Spotlight, type `Terminal`,
  and press Enter.

You will see a window with some text and a blinking cursor waiting
for input. That is normal, it is just waiting for you to type a
command and press Enter.

A few things worth knowing before you start:

- **A command is a line of text you type, followed by Enter.** After
  you press Enter, the computer runs it. While it is running, you
  may see text scroll by, or nothing at all, both are normal. When
  it is done, you will see the cursor and prompt again, ready for
  the next command.
- **"Navigating to a folder" means telling the terminal which folder
  to work in**, using the `cd` (change directory) command, for
  example `cd Documents`. All the commands in this guide should be
  run from inside the project folder once you have cloned it (step 1
  below explains this).
- **Copy and paste commands exactly as written**, including
  capitalization and punctuation. Terminals are strict about this.
- If a command seems to hang for a long time with no output, that is
  often normal for downloads or installs, see the "what to expect"
  notes under each step.

## Prerequisites

Before starting, make sure you have the following installed. Each
one includes a quick way to check whether it is already on your
computer.

- **Python 3.12** installed ([python.org/downloads](https://www.python.org/downloads/))
  - Python is the programming language this project is written in.
    You need it installed so your computer can run the project's
    scripts.
  - To check if you already have it, open a terminal (see above) and
    run `python --version`. If you see something like
    `Python 3.12.1`, you are set. If you see an error like
    "command not found", you need to install it.
  - **Windows only**: during installation, make sure to check the
    box labeled **"Add Python to PATH"** on the first install
    screen. If you miss this, `python` commands will not work later,
    see Troubleshooting.
- **Git** installed
  - Git is the tool used to download ("clone") this project's code
    from GitHub to your computer.
  - Check with `git --version`. If it prints a version number, you
    are set. Otherwise install it from
    [git-scm.com/downloads](https://git-scm.com/downloads).
- **wget** installed (used to download one of the data sources)
  - wget is a small tool that downloads files from the internet on
    your behalf, one of the setup scripts uses it automatically.
  - Windows: `winget install JernejSimoncic.Wget` (or `choco install wget`),
    then close and reopen your terminal
  - Mac: `brew install wget`
  - Linux: usually pre-installed, otherwise `apt install wget`
  - Check with `wget --version` after installing.
- A free **Kaggle account** (needed for one data source, see setup below)
  - Kaggle is a website that hosts one of the datasets this project
    uses. You will need a free account and an API key, walked
    through in step 3 below.

## 1. Clone the repository

"Cloning" means downloading a full copy of this project's code and
files onto your computer, into a new folder that git creates for
you.

```bash
git clone https://github.com/PrincewillDev/Healthcare-Phishing-Detection-System.git
cd Healthcare-Phishing-Detection-System
```

The first command downloads the project. The second command (`cd`,
"change directory") moves your terminal into the new project folder,
which git names `Healthcare-Phishing-Detection-System` by default.

**What success looks like:** the `git clone` command prints a few
lines about "Cloning into..." and "Receiving objects...", then
returns you to the prompt. A new folder called
`Healthcare-Phishing-Detection-System` will appear wherever you ran
the command from. After running `cd`, your terminal prompt will
usually show the folder name, confirming you are now inside it. All
remaining commands in this guide should be run from inside this
folder.

## 2. Set up the Python environment

A "virtual environment" is an isolated folder where this project's
Python packages get installed, separate from any other Python
projects on your computer. This avoids version conflicts between
projects. Think of it as a dedicated toolbox just for this project.

```bash
python -m venv venv
```

This creates the virtual environment. It does not print much output.

**What success looks like:** a new folder named `venv` appears
inside the project folder. This may take a few seconds.

Activate it (this tells your terminal to start using the tools
inside that `venv` folder instead of your computer's general Python
setup):

- **Windows (PowerShell)**: `venv\Scripts\Activate.ps1`
- **Mac/Linux**: `source venv/bin/activate`

**What success looks like:** your terminal prompt changes to start
with `(venv)`, for example `(venv) PS C:\...>`. This tells you the
virtual environment is active. You will need to activate it again
this way each time you open a new terminal window to work on this
project.

Then install the project's dependencies (the external Python
packages this project relies on, such as machine learning and data
libraries):

```bash
pip install -r requirements.txt
```

**What success looks like:** you will see a long list of package
names scroll by as they download and install. This can take a few
minutes depending on your internet connection. It is done when the
prompt returns and the last few lines do not say `ERROR`.

## 3. Set up Kaggle API credentials

One data source is hosted on Kaggle and requires an API key (a
personal access code that lets the project's scripts download data
on your behalf, without you having to do it manually).

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings)
   (log in or create a free account first if you do not have one)
2. Scroll to the **API** section
3. Click **Create New Token**, this downloads `kaggle.json`, a small
   file containing your personal API key. Treat it like a password,
   do not share it or post it publicly.
4. Move it to the correct location, so the project's scripts can
   find it automatically:

   **Windows (PowerShell)**:
   ```powershell
   mkdir $env:USERPROFILE\.kaggle -ErrorAction SilentlyContinue
   move $env:USERPROFILE\Downloads\kaggle.json $env:USERPROFILE\.kaggle\kaggle.json
   ```

   **Mac/Linux**:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```

   These commands assume the file downloaded to your default
   Downloads folder. If you saved it somewhere else, adjust the path
   accordingly.

5. Verify it's in place:
   ```bash
   # Windows
   type $env:USERPROFILE\.kaggle\kaggle.json
   # Mac/Linux
   cat ~/.kaggle/kaggle.json
   ```
   You should see `{"username":"...","key":"..."}`. If instead you
   see an error like "file not found", the file was not moved to
   the right place, repeat step 4.

## 4. Run the data pipeline

A "pipeline" here means a sequence of scripts that automatically
download and process the data this project needs, in order. You
run each one as a command and wait for it to finish before running
the next.

Everything runs in order, each step is idempotent (safe to re-run,
it will skip anything already downloaded or generated, so it is
safe to rerun a step if it gets interrupted).

```bash
# Step 1: Download raw data sources (SpamAssassin, Nazario, Kaggle)
python src/preprocessing/run_data_pipeline.py

# Step 2: Clean the merged dataset
python src/preprocessing/clean_dataset.py --apply-non-email-filter --write

# Step 3: Sample to final dataset and create train/val/test splits
python src/preprocessing/sample_and_split.py
```

- **Step 1** downloads roughly 41,000 raw emails from three
  different sources onto your computer. Expect this to take a while
  and to print a lot of progress messages, that is normal, let it
  run until the prompt returns on its own.
- **Step 2** removes duplicate, blank, and non-email content from
  the downloaded data, and prints a report of what it removed.
- **Step 3** selects the final set of emails used for training and
  splits them into training, validation, and test groups, then
  prints a summary of the counts.

Steps 2 and 3 run in under a minute each.

**What success looks like:** each command prints a report as it
runs (counts, tables, summaries) and returns you to the prompt
without an error at the end. After all three steps finish, a `data`
folder will contain the downloaded and processed files described
below.

## What you'll have after running this

```
data/
├── raw/                          # untouched source data (not tracked in git)
│   ├── spamassassin.csv
│   ├── nazario/
│   └── healthcare_phishing.csv
├── synthetic/                    # hand-authored healthcare-context emails (tracked in git)
│   └── healthcare_synthetic_batch*.csv
└── processed/                    # not tracked in git, regenerated by the pipeline
    ├── merged_raw.csv
    ├── cleaned.csv
    ├── train.csv                 # 10,500 rows
    ├── val.csv                   # 2,250 rows
    ├── test.csv                  # 2,250 rows
    └── confound_check.csv        # 1,000 rows, diagnostic set
```

## Dataset summary

| Source | Role | Notes |
|---|---|---|
| SpamAssassin | Legitimate email baseline | Spam-labeled content excluded from training |
| Nazario | Real phishing emails | Full headers preserved |
| Kaggle (Chakraborty) | General phishing/legitimate volume | Not healthcare-specific |
| Synthetic (self-authored) | Healthcare-context phishing/legitimate | 1,000 emails, 20 subcategories, since no public healthcare-specific phishing dataset exists |

Final training dataset: **15,000 emails, 50/50 phishing/legitimate**,
split 70/15/15 into train/validation/test.

## Known limitations (documented, not blockers)

- 512 Nazario rows contain characters that couldn't be recovered
  during encoding correction (original source data limitation)
- The confound-check diagnostic set is sourced entirely from Kaggle
  (the synthetic healthcare set was fully consumed by the main
  15,000-row sample)

## Troubleshooting

**"python is not recognized as an internal or external command" (Windows) or "command not found" (Mac)**
Python is either not installed or not added to your system's PATH
(the list of places your computer looks for programs to run). On
Windows, reinstall Python from
[python.org/downloads](https://www.python.org/downloads/) and make
sure to check "Add Python to PATH" during setup. Close and reopen
your terminal afterward, PATH changes only take effect in new
terminal windows.

**I have Python, but it's the wrong version**
Run `python --version` to see what you have. This project expects
Python 3.12. If your computer has multiple versions installed, try
`python3 --version` or `py --version` (Windows) to see if a
different command points to the right one, and use that command
throughout this guide instead of `python`.

**`pip install -r requirements.txt` fails partway through**
- If you see network-related errors, check your internet connection
  and try running the command again.
- If you see permission errors, make sure you activated the virtual
  environment first (step 2), you should see `(venv)` at the start
  of your prompt. Installing without an active virtual environment
  can require extra permissions you may not have.
- If a specific package fails to build, note its name and search for
  that exact error message, some packages need extra system tools
  that vary by operating system.

**Kaggle authentication errors when running Step 1 of the pipeline**
This means `kaggle.json` was not found or not set up correctly.
Revisit step 3 above:
- Confirm the file exists at `~/.kaggle/kaggle.json` (Mac/Linux) or
  `%USERPROFILE%\.kaggle\kaggle.json` (Windows) using the "verify"
  command in step 3.
- Confirm it contains your actual username and key, not placeholder
  text, by opening it with the "verify" command.
- If you downloaded a new token from Kaggle, make sure you moved the
  newest one, old tokens can be invalidated when you create a new one.

**"wget is not recognized" / "command not found: wget"**
wget was not installed, or your terminal was not restarted after
installing it. Revisit the Prerequisites section, install wget for
your operating system, then fully close and reopen your terminal
before trying again.

**A command seems stuck with no output for a long time**
For Step 1 of the data pipeline especially, this can be normal, it
is downloading a large amount of data and may pause between sources.
Give it several minutes before assuming something is wrong. If it
truly hangs (many minutes with no change and no network activity),
press `Ctrl + C` to stop it and try running the same command again,
the pipeline is safe to re-run.

**General permission errors ("Access is denied", "Permission denied")**
Make sure you are not running commands inside a system-protected
folder (for example, directly inside `Program Files` on Windows).
Clone the repository into a normal location like your Documents
folder instead. On Mac/Linux, avoid using `sudo` with these
commands, if you are prompted to use it, that usually signals the
virtual environment was not activated correctly.

## Project status

Data preparation phase complete. Next phase: feature engineering
(header analysis, text/TF-IDF features, URL/domain features).
