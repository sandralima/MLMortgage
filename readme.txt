# to install cookiecutter make sure that anaconda prompt had started as Administrator.
$ conda config --add channels conda-forge
$ conda install cookiecutter

# in the working directory:
$ cd /C/Users/sandr/Documents/ENN/DLMortgage/DLMortgage/DLMortCookiecutter
$ git clone git@github.com:audreyr/cookiecutter-pypackage.git

#if there is trouble because you don't have git credentials in your local machine, you must be to follow the next steps (into the git bash previously installed):

$ ssh-keygen -t rsa -b 4096 -C sandralima111@gmail.com -f/C/Users/sandr/Documents/github_rsakey
Generating public/private rsa key pair.
Enter passphrase (empty for no passphrase):
Enter same passphrase again:
Your identification has been saved in /C/Users/sandr/Documents/github_rsakey.
Your public key has been saved in /C/Users/sandr/Documents/github_rsakey.pub.
The key fingerprint is:
SHA256:+2C0en39nSuWYzmu8KC0LWpFZiCvlj+mwsnTD7/H4+Q sandralima111@gmail.com
The key's randomart image is:
+---[RSA 4096]----+
|                 |
|   . .           |
|    o .          |
|     . +         |
|    o + S        |
|   +   o o       |
|o +.. oo=+   .o  |
| * .o=+B=o= .O. o|
|  o.==BEo..+=.+++|
+----[SHA256]-----+


$ eval $(ssh-agent -s)
Agent pid 10336

$ ssh-add /C/Users/sandr/Documents/github_rsakey
Identity added: /C/Users/sandr/Documents/github_rsakey (/C/Users/sandr/Documents/github_rsakey)

$ ssh -T git@github.com
The authenticity can't be stablished
git@github.com: Permission denied (publickey).


# in the working directory let's execute:
$ eval $(ssh-agent -s)
Agent pid 10336

$ ssh-add /C/Users/sandr/Documents/github_rsakey
Identity added: /C/Users/sandr/Documents/github_rsakey (/C/Users/sandr/Documents/github_rsakey)

# but the same error:
$ ssh -T git@github.com
The authenticity can't be stablished
git@github.com: Permission denied (publickey).

# let's try with a git shell executed as administrator:
$ ssh-keygen -t rsa -b 4096 -C sandralima111@gmail.com -f/C/Users/sandr/Documents/github_rsakey
Generating public/private rsa key pair.
/C/Users/sandr/Documents/github_rsakey already exists.
Overwrite (y/n)? y
Enter passphrase (empty for no passphrase):
Enter same passphrase again:
Your identification has been saved in /C/Users/sandr/Documents/github_rsakey.
Your public key has been saved in /C/Users/sandr/Documents/github_rsakey.pub.
The key fingerprint is:
SHA256:PyoZc3VwuPzxfpM0R21Chr5hP17ViDiUUS5pCIrnXgg sandralima111@gmail.com
The key's randomart image is:
+---[RSA 4096]----+
|      .   .=..   |
|   . . . .=oo o  |
|  E o   .o+*.+ .o|
|   + .   .*.B o *|
|    o . S. = * +.|
|   . .o ..  o +oo|
|    .  =  o  o.o+|
|      o  . .  oo.|
|       ..      ..|
+----[SHA256]-----+

$ ssh-add -l
4096 SHA256:PyoZc3VwuPzxfpM0R21Chr5hP17ViDiUUS5pCIrnXgg /C/Users/sandr/Documents/github_rsakey (RSA)

$ ssh -vT git@github.com
OpenSSH_7.6p1, OpenSSL 1.0.2n  7 Dec 2017
debug1: Reading configuration data /etc/ssh/ssh_config
debug1: Connecting to github.com [192.30.253.112] port 22.
debug1: Connection established.
debug1: key_load_public: No such file or directory
debug1: identity file /c/Users/sandr/.ssh/id_rsa type -1
debug1: key_load_public: No such file or directory
debug1: identity file /c/Users/sandr/.ssh/id_rsa-cert type -1
debug1: key_load_public: No such file or directory
debug1: identity file /c/Users/sandr/.ssh/id_dsa type -1
debug1: key_load_public: No such file or directory
debug1: identity file /c/Users/sandr/.ssh/id_dsa-cert type -1
debug1: key_load_public: No such file or directory
debug1: identity file /c/Users/sandr/.ssh/id_ecdsa type -1
debug1: key_load_public: No such file or directory
debug1: identity file /c/Users/sandr/.ssh/id_ecdsa-cert type -1
debug1: key_load_public: No such file or directory
debug1: identity file /c/Users/sandr/.ssh/id_ed25519 type -1
debug1: key_load_public: No such file or directory
debug1: identity file /c/Users/sandr/.ssh/id_ed25519-cert type -1
debug1: Local version string SSH-2.0-OpenSSH_7.6
debug1: Remote protocol version 2.0, remote software version libssh_0.7.0
debug1: no match: libssh_0.7.0
debug1: Authenticating to github.com:22 as 'git'
debug1: SSH2_MSG_KEXINIT sent
debug1: SSH2_MSG_KEXINIT received
debug1: kex: algorithm: curve25519-sha256@libssh.org
debug1: kex: host key algorithm: ssh-rsa
debug1: kex: server->client cipher: aes128-ctr MAC: hmac-sha2-256 compression: none
debug1: kex: client->server cipher: aes128-ctr MAC: hmac-sha2-256 compression: none
debug1: expecting SSH2_MSG_KEX_ECDH_REPLY
debug1: Server host key: ssh-rsa SHA256:nThbg6kXUpJWGl7E1IGOCspRomTxdCARLviKw6E5SY8
debug1: Host 'github.com' is known and matches the RSA host key.
debug1: Found key in /c/Users/sandr/.ssh/known_hosts:1
debug1: rekey after 4294967296 blocks
debug1: SSH2_MSG_NEWKEYS sent
debug1: expecting SSH2_MSG_NEWKEYS
debug1: SSH2_MSG_NEWKEYS received
debug1: rekey after 4294967296 blocks
debug1: SSH2_MSG_SERVICE_ACCEPT received
debug1: Authentications that can continue: publickey
debug1: Next authentication method: publickey
debug1: Offering public key: RSA SHA256:PyoZc3VwuPzxfpM0R21Chr5hP17ViDiUUS5pCIrnXgg /C/Users/sandr/Documents/github_rsakey
debug1: Authentications that can continue: publickey
debug1: Trying private key: /c/Users/sandr/.ssh/id_rsa
debug1: Trying private key: /c/Users/sandr/.ssh/id_dsa
debug1: Trying private key: /c/Users/sandr/.ssh/id_ecdsa
debug1: Trying private key: /c/Users/sandr/.ssh/id_ed25519
debug1: No more authentication methods to try.
git@github.com: Permission denied (publickey).

# we need to put the rsakey in directory /c/Users/sandr/.ssh/:
 ssh-keygen -t rsa -b 4096 -C sandralima111@gmail.com -f/C/Users/sandr/.ssh/id_rsa
Generating public/private rsa key pair.
/C/Users/sandr/.ssh/id_rsa already exists.
Overwrite (y/n)? y
Enter passphrase (empty for no passphrase):
Enter same passphrase again:
Your identification has been saved in /C/Users/sandr/.ssh/id_rsa.
Your public key has been saved in /C/Users/sandr/.ssh/id_rsa.pub.
The key fingerprint is:
SHA256:0wA10zZ4TylndNEXLiFRtffPtN8ImOyjC4DU2mu9Loc sandralima111@gmail.com
The key's randomart image is:
+---[RSA 4096]----+
|      ..+o ++++=.|
|   .   ..o* *.o +|
|  . .   .o B . oo|
| . +     o  . . o|
|  o o   S .     o|
|     +   o o   oo|
|    o.o   + .  .o|
|   .E .o ..  . .o|
|     +o oo..  . o|
+----[SHA256]-----+


$ ssh-add /C/Users/sandr/.ssh/id_rsa
Identity added: /C/Users/sandr/.ssh/id_rsa (/C/Users/sandr/.ssh/id_rsa)

sandr@DESKTOP-2P2FJVH MINGW64 /C/Users/sandr/Documents/ENN/DLMortgage/DLMortgage/DLMortCookiecutter
$ ssh-add -l
4096 SHA256:PyoZc3VwuPzxfpM0R21Chr5hP17ViDiUUS5pCIrnXgg /C/Users/sandr/Documents/github_rsakey (RSA)
4096 SHA256:0wA10zZ4TylndNEXLiFRtffPtN8ImOyjC4DU2mu9Loc /C/Users/sandr/.ssh/id_rsa (RSA)

# to add your rsa_key.pub to your profile in github account --> settings --> SSH and GPG keys
$ clip < /C/Users/sandr/.ssh/id_rsa.pub
# Copies the contents of the id_rsa.pub file to your clipboard

sandr@DESKTOP-2P2FJVH MINGW64 /C/Users/sandr/Documents/ENN/DLMortgage/DLMortgage/DLMortCookiecutter
$ ssh-add -l -E md5
The agent has no identities.

sandr@DESKTOP-2P2FJVH MINGW64 /C/Users/sandr/Documents/ENN/DLMortgage/DLMortgage/DLMortCookiecutter
$ ssh-add -l
The agent has no identities.

sandr@DESKTOP-2P2FJVH MINGW64 /C/Users/sandr/Documents/ENN/DLMortgage/DLMortgage/DLMortCookiecutter
$ eval $(ssh-agent -s)
Agent pid 10104

sandr@DESKTOP-2P2FJVH MINGW64 /C/Users/sandr/Documents/ENN/DLMortgage/DLMortgage/DLMortCookiecutter
$ ssh-add -l
The agent has no identities.

sandr@DESKTOP-2P2FJVH MINGW64 /C/Users/sandr/Documents/ENN/DLMortgage/DLMortgage/DLMortCookiecutter
$ ssh-add /C/Users/sandr/.ssh/id_rsa
Identity added: /C/Users/sandr/.ssh/id_rsa (/C/Users/sandr/.ssh/id_rsa)

sandr@DESKTOP-2P2FJVH MINGW64 /C/Users/sandr/Documents/ENN/DLMortgage/DLMortgage/DLMortCookiecutter
$ eval $(ssh-agent -s)
Agent pid 12048

sandr@DESKTOP-2P2FJVH MINGW64 /C/Users/sandr/Documents/ENN/DLMortgage/DLMortgage/DLMortCookiecutter
$ ssh-add -l
The agent has no identities.

sandr@DESKTOP-2P2FJVH MINGW64 /C/Users/sandr/Documents/ENN/DLMortgage/DLMortgage/DLMortCookiecutter
$ eval $(ssh-agent -s)
Agent pid 14152

sandr@DESKTOP-2P2FJVH MINGW64 /C/Users/sandr/Documents/ENN/DLMortgage/DLMortgage/DLMortCookiecutter
$ ssh-add /C/Users/sandr/.ssh/id_rsa
Identity added: /C/Users/sandr/.ssh/id_rsa (/C/Users/sandr/.ssh/id_rsa)

sandr@DESKTOP-2P2FJVH MINGW64 /C/Users/sandr/Documents/ENN/DLMortgage/DLMortgage/DLMortCookiecutter
$ ssh-add -l
4096 SHA256:0wA10zZ4TylndNEXLiFRtffPtN8ImOyjC4DU2mu9Loc /C/Users/sandr/.ssh/id_rsa (RSA)

sandr@DESKTOP-2P2FJVH MINGW64 /C/Users/sandr/Documents/ENN/DLMortgage/DLMortgage/DLMortCookiecutter
$ ssh-add -l -E md5
4096 MD5:98:79:8e:52:e9:1c:9e:ca:a5:4b:0a:bf:b8:87:3b:f1 /C/Users/sandr/.ssh/id_rsa (RSA)
# this is equal to the keygen saved in the github account settings:
sshKeyASUSLaptop
Fingerprint: 98:79:8e:52:e9:1c:9e:ca:a5:4b:0a:bf:b8:87:3b:f1 Added on Feb 27, 2018 Never used — Read/write 

sandr@DESKTOP-2P2FJVH MINGW64 /C/Users/sandr/Documents/ENN/DLMortgage/DLMortgage/DLMortCookiecutter
$ ssh -vT git@github.com
OpenSSH_7.6p1, OpenSSL 1.0.2n  7 Dec 2017
debug1: Reading configuration data /etc/ssh/ssh_config
debug1: Connecting to github.com [192.30.253.112] port 22.
debug1: Connection established.
debug1: identity file /c/Users/sandr/.ssh/id_rsa type 0
debug1: key_load_public: No such file or directory
debug1: identity file /c/Users/sandr/.ssh/id_rsa-cert type -1
debug1: key_load_public: No such file or directory
debug1: identity file /c/Users/sandr/.ssh/id_dsa type -1
debug1: key_load_public: No such file or directory
debug1: identity file /c/Users/sandr/.ssh/id_dsa-cert type -1
debug1: key_load_public: No such file or directory
debug1: identity file /c/Users/sandr/.ssh/id_ecdsa type -1
debug1: key_load_public: No such file or directory
debug1: identity file /c/Users/sandr/.ssh/id_ecdsa-cert type -1
debug1: key_load_public: No such file or directory
debug1: identity file /c/Users/sandr/.ssh/id_ed25519 type -1
debug1: key_load_public: No such file or directory
debug1: identity file /c/Users/sandr/.ssh/id_ed25519-cert type -1
debug1: Local version string SSH-2.0-OpenSSH_7.6
debug1: Remote protocol version 2.0, remote software version libssh_0.7.0
debug1: no match: libssh_0.7.0
debug1: Authenticating to github.com:22 as 'git'
debug1: SSH2_MSG_KEXINIT sent
debug1: SSH2_MSG_KEXINIT received
debug1: kex: algorithm: curve25519-sha256@libssh.org
debug1: kex: host key algorithm: ssh-rsa
debug1: kex: server->client cipher: aes128-ctr MAC: hmac-sha2-256 compression: none
debug1: kex: client->server cipher: aes128-ctr MAC: hmac-sha2-256 compression: none
debug1: expecting SSH2_MSG_KEX_ECDH_REPLY
debug1: Server host key: ssh-rsa SHA256:nThbg6kXUpJWGl7E1IGOCspRomTxdCARLviKw6E5SY8
debug1: Host 'github.com' is known and matches the RSA host key.
debug1: Found key in /c/Users/sandr/.ssh/known_hosts:1
debug1: rekey after 4294967296 blocks
debug1: SSH2_MSG_NEWKEYS sent
debug1: expecting SSH2_MSG_NEWKEYS
debug1: SSH2_MSG_NEWKEYS received
debug1: rekey after 4294967296 blocks
debug1: SSH2_MSG_SERVICE_ACCEPT received
debug1: Authentications that can continue: publickey
debug1: Next authentication method: publickey
debug1: Offering public key: RSA SHA256:0wA10zZ4TylndNEXLiFRtffPtN8ImOyjC4DU2mu9Loc /c/Users/sandr/.ssh/id_rsa
debug1: Server accepts key: pkalg ssh-rsa blen 535
debug1: Authentication succeeded (publickey).
Authenticated to github.com ([192.30.253.112]:22).
debug1: channel 0: new [client-session]
debug1: Entering interactive session.
debug1: pledge: network
Hi sandralima! You've successfully authenticated, but GitHub does not provide shell access.
debug1: client_input_channel_req: channel 0 rtype exit-status reply 0
debug1: channel 0: free: client-session, nchannels 1
Transferred: sent 3328, received 2072 bytes, in 0.2 seconds
Bytes per second: sent 17461.4, received 10871.4
debug1: Exit status 1

# some issues about ssh keygen:
https://help.github.com/articles/error-permission-denied-publickey/

sandr@DESKTOP-2P2FJVH MINGW64 /C/Users/sandr/Documents/ENN/DLMortgage/DLMortgage/DLMortCookiecutter
$ git clone git@github.com:audreyr/cookiecutter-pypackage.git
Cloning into 'cookiecutter-pypackage'...
remote: Counting objects: 2409, done.
remote: Compressing objects: 100% (14/14), done.
Receivinremote: Total 2409 (delta 5), reused 0 (delta 0), pack-reused 2395
Receiving objects: 100% (2409/2409), 392.80 KiB | 1.71 MiB/s, done.
Resolving deltas: 100% (1443/1443), done.
#the template is done!!

sandr@DESKTOP-2P2FJVH MINGW64 /C/Users/sandr/Documents/ENN/DLMortgage/DLMortgage/DLMortCookiecutter
$ cookiecutter cookiecutter-pypackage/
bash: cookiecutter: command not found

# this command must be executed from anaconda (Administrator) Prompt:
(base) C:\Users\sandr\Documents\ENN\DLMortgage\DLMortgage\DLMortCookiecutter>cookiecutter cookiecutter-pypackage/
full_name [Sandra Lima]:
email [sandralima111@gmail.com]:
github_username [sandralima]:
project_name [MLMortgage]:
project_slug [mlmortgage]:
project_short_description [Machine Learning for Mortgages.]:
Unable to render variable 'pypi_username'
Error message: 'MLMortgage' is undefined
Context: {
    "cookiecutter": {
        "add_pyup_badge": "n",
        "command_line_interface": [
            "Click",
            "No command-line interface"
        ],
        "create_author_file": "y",
        "email": "sandralima111@gmail.com",
        "full_name": "Sandra Lima",
        "github_username": "sandralima",
        "open_source_license": [
            "MIT license",
            "BSD license",
            "ISC license",
            "Apache Software License 2.0",
            "GNU General Public License v3",
            "Not open source"
        ],
        "project_name": "MLMortgage",
        "project_short_description": "Machine Learning for Mortgages.",
        "project_slug": "{{ cookiecutter.project_name.lower().replace(' ', '_').replace('-', '_') }}",
        "pypi_username": "{{ MLMortgage.github_username }}",
        "use_pypi_deployment_with_travis": "y",
        "use_pytest": "n",
        "version": "0.1.0"
    }
}

# Trying through a different command to put the values directly in the prompt:
(base) C:\Users\sandr\Documents\ENN\DLMortgage\DLMortgage\DLMortCookiecutter\other>cookiecutter https://github.com/audreyr/cookiecutter-pypackage.git
full_name [Audrey Roy Greenfeld]: Sandra Lima
email [aroy@alum.mit.edu]: sandralima111@gmail.com
github_username [audreyr]: sandralima
project_name [Python Boilerplate]: MLMortgage
project_slug [mlmortgage]:
project_short_description [Python Boilerplate contains all the boilerplate you need to create a Python package.]: Machine Learning for Mortgages.
pypi_username [sandralima]:
version [0.1.0]:
use_pytest [n]:
use_pypi_deployment_with_travis [y]:
add_pyup_badge [n]:
Select command_line_interface:
1 - Click
2 - No command-line interface
Choose from 1, 2 [1]:
create_author_file [y]:
Select open_source_license:
1 - MIT license
2 - BSD license
3 - ISC license
4 - Apache Software License 2.0
5 - GNU General Public License v3
6 - Not open source
Choose from 1, 2, 3, 4, 5, 6 [1]:

(base) C:\Users\sandr\Documents\ENN\DLMortgage\DLMortgage\DLMortCookiecutter\other>tree
Folder PATH listing for volume OS
Volume serial number is 9408-3673
C:.
+---other
    +---mlmortgage
        +---.github
        +---docs
        +---mlmortgage
        +---tests

Whenever you generate a project with a cookiecutter, the resulting project is output to your current directory.
Your cloned cookiecutters are stored by default in your ~/.cookiecutters/ directory (or Windows equivalent). 

But this is the basic cookiecutter python project. For data science:

$ cookiecutter https://github.com/drivendata/cookiecutter-data-science
project_name [project_name]: MLMortgage
repo_name [mlmortgage]:
author_name [Your name (or your organization/company/team)]: Sandra Lima
description [A short description of the project.]: Machine Learning for mortgages
Select open_source_license:
1 - MIT
2 - BSD
3 - Not open source
Choose from 1, 2, 3 [1]:
s3_bucket [[OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')]:
aws_profile [default]:
Select python_interpreter:
1 - python
2 - python3
Choose from 1, 2 [1]: 2

(base) C:\Users\sandr\Documents\ENN\DLMortgage\DLMortgage\DLMortCookiecutter\mlmortgage>tree
Folder PATH listing for volume OS
Volume serial number is 9408-3673
C:.
+---data
¦   +---external
¦   +---interim
¦   +---processed
¦   +---raw
+---docs
+---models
+---notebooks
+---references
+---reports
¦   +---figures
+---src
    +---data
    +---features
    +---models
    +---visualization

# GITHUB REPOSITORY:

Let's execute the following commands from GIT bash (GIT command prompt):
sandr@DESKTOP-2P2FJVH MINGW64 /C/Users/sandr/Documents/ENN/DLMortgage/DLMortgage/DLMortCookiecutter/mlmortgage
$ git init
Initialized empty Git repository in C:/Users/sandr/Documents/ENN/DLMortgage/DLMortgage/DLMortCookiecutter/mlmortgage/.git/

sandr@DESKTOP-2P2FJVH MINGW64 /C/Users/sandr/Documents/ENN/DLMortgage/DLMortgage/DLMortCookiecutter/mlmortgage (master)
$ git add .
# to add all files (.) to the repository.

sandr@DESKTOP-2P2FJVH MINGW64 /C/Users/sandr/Documents/ENN/DLMortgage/DLMortgage/DLMortCookiecutter/mlmortgage (master)
$ git commit -m 'initial commit'

$git push

*** Please tell me who you are.

Run

  git config --global user.email "you@example.com"
  git config --global user.name "Your Name"

to set your account's default identity.
Omit --global to set the identity only in this repository.

fatal: unable to auto-detect email address (got 'sandr@DESKTOP-2P2FJVH.(none)')

sandr@DESKTOP-2P2FJVH MINGW64 /C/Users/sandr/Documents/ENN/DLMortgage/DLMortgage/DLMortCookiecutter/mlmortgage (master)
$ git config --global sandralima111@gmail.com
error: invalid key: sandralima111@gmail.com

sandr@DESKTOP-2P2FJVH MINGW64 /C/Users/sandr/Documents/ENN/DLMortgage/DLMortgage/DLMortCookiecutter/mlmortgage (master)
$ git config --global user.email "sandralima111@gmail.com"

sandr@DESKTOP-2P2FJVH MINGW64 /C/Users/sandr/Documents/ENN/DLMortgage/DLMortgage/DLMortCookiecutter/mlmortgage (master)
$ git commit -m 'initial commit'
[master (root-commit) 840b49b] initial commit
 28 files changed, 980 insertions(+)
 create mode 100644 .gitignore
 create mode 100644 LICENSE
 create mode 100644 Makefile
 create mode 100644 README.md
 create mode 100644 docs/Makefile
 create mode 100644 docs/commands.rst
 create mode 100644 docs/conf.py
 create mode 100644 docs/getting-started.rst
 create mode 100644 docs/index.rst
 create mode 100644 docs/make.bat
 create mode 100644 models/.gitkeep
 create mode 100644 notebooks/.gitkeep
 create mode 100644 references/.gitkeep
 create mode 100644 reports/.gitkeep
 create mode 100644 reports/figures/.gitkeep
 create mode 100644 requirements.txt
 create mode 100644 src/__init__.py
 create mode 100644 src/data/.gitkeep
 create mode 100644 src/data/make_dataset.py
 create mode 100644 src/features/.gitkeep
 create mode 100644 src/features/build_features.py
 create mode 100644 src/models/.gitkeep
 create mode 100644 src/models/predict_model.py
 create mode 100644 src/models/train_model.py
 create mode 100644 src/visualization/.gitkeep
 create mode 100644 src/visualization/visualize.py
 create mode 100644 test_environment.py

sandr@DESKTOP-2P2FJVH MINGW64 /C/Users/sandr/Documents/ENN/DLMortgage/DLMortgage/DLMortCookiecutter/mlmortgage (master)
$ git log
commit 840b49b19cb89ec4bda91a8f195c7e1f36b68850 (HEAD -> master)
Author: Sandra Nataly Lima Castro <sandralima111@gmail.com>
Date:   Tue Feb 27 15:17:32 2018 -0500

    initial commit

sandr@DESKTOP-2P2FJVH MINGW64 /C/Users/sandr/Documents/ENN/DLMortgage/DLMortgage/DLMortCookiecutter/mlmortgage (master)
$ git log --oneline
840b49b (HEAD -> master) initial commit

# all of this is done in a local level, to update the changes over a remote repository, you must be created this repo at the webpage and then to allow to git command to acknowledge the remote repo through the following command:

$ git remote add origin https://github.com/username/myproject.git

# you must replace username and myproject with whatever your GITHUB username and project actually are.

$ git remote add origin https://github.com/sandralima/MLMortgage.git

sandr@DESKTOP-2P2FJVH MINGW64 /C/Users/sandr/Documents/ENN/DLMortgage/DLMortgage/DLMortCookiecutter/mlmortgage (master)
$ git remote -v
origin  https://github.com/sandralima/MLMortgage.git (fetch)
origin  https://github.com/sandralima/MLMortgage.git (push)

sandr@DESKTOP-2P2FJVH MINGW64 /C/Users/sandr/Documents/ENN/DLMortgage/DLMortgage/DLMortCookiecutter/mlmortgage (master)
$ git push origin master
Counting objects: 25, done.
Delta compression using up to 4 threads.
Compressing objects: 100% (23/23), done.
Writing objects: 100% (25/25), 12.41 KiB | 1.55 MiB/s, done.
Total 25 (delta 0), reused 0 (delta 0)
To https://github.com/sandralima/MLMortgage.git
 * [new branch]      master -> master

sandr@DESKTOP-2P2FJVH MINGW64 /C/Users/sandr/Documents/ENN/DLMortgage/DLMortgage/DLMortCookiecutter/mlmortgage (master)
$ git status
On branch master
nothing to commit, working tree clean

# from an example:
git add chocolate.jpeg

Now, take a “snapshot” of the repository as it stands now with the commit command:

git commit -m “Add chocolate.jpeg.”

# drivendata project:
https://github.com/drivendata/cookiecutter-data-science

# to begin with make_dataset.py:

installing some useful packages:

$ conda install --name tensorflowenvironment requests
$ conda install --name tensorflowenvironment beautifulsoup4

for an example with request:
{
  "tone_analyzer": [
    {
      "name": "tone-analyzer-natural-la-tone-analy-1519832806660",
      "plan": "lite",
      "credentials": {
        "url": "https://gateway.watsonplatform.net/tone-analyzer/api",
        "username": "e662ea83-7e4b-4bc8-b573-9823220bd3e0",
        "password": "qJfyIIne8IMp"
      }
    }
  ],
  "natural-language-understanding": [
    {
      "name": "tone-analyzer-natural-la-natural-la-1519832806656",
      "plan": "free",
      "credentials": {
        "url": "https://gateway.watsonplatform.net/natural-language-understanding/api",
        "username": "c36dd57c-5fca-4502-8746-0af9e75ae742",
        "password": "ocsYZZWBvWGE"
      }
    }
  ]
}

but this conexion works with watson ID and other private packages... then I dont matter.

$ conda install --name tensorflowenvironment -y -q pymysql
$ conda install --name tensorflowenvironment -y -q python-dotenv

#to access to kaggle:
Ensure kaggle.json is in the location ~/.kaggle/kaggle.json to use the API.

# to commit all changes to git repository (from git prompt run as administrator):
 cd /C/Users/sandr/Documents/personal/titanic

sandr@DESKTOP-2P2FJVH MINGW64 /C/Users/sandr/Documents/personal/titanic (master)
$ git status
On branch master
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

        modified:   src/data/make_dataset.py

Untracked files:
  (use "git add <file>..." to include in what will be committed)

        notebooks/make_dataset_test.ipynb
        notebooks/making_data.ipynb
        src/data/get_raw_data.py

no changes added to commit (use "git add" and/or "git commit -a")

sandr@DESKTOP-2P2FJVH MINGW64 /C/Users/sandr/Documents/personal/titanic (master)
$ git log --oneline
be7e37e (HEAD -> master, origin/master) initial commit

$ git add .
warning: LF will be replaced by CRLF in notebooks/make_dataset_test.ipynb.
The file will have its original line endings in your working directory.
warning: LF will be replaced by CRLF in notebooks/making_data.ipynb.
The file will have its original line endings in your working directory.

sandr@DESKTOP-2P2FJVH MINGW64 /C/Users/sandr/Documents/personal/titanic (master)
$ git commit -m "data extraction from Kaggle"
[master de5ce6f] data extraction from Kaggle
 4 files changed, 1316 insertions(+), 2 deletions(-)
 create mode 100644 notebooks/make_dataset_test.ipynb
 create mode 100644 notebooks/making_data.ipynb
 create mode 100644 src/data/get_raw_data.py

sandr@DESKTOP-2P2FJVH MINGW64 /C/Users/sandr/Documents/personal/titanic (master)
$ git log
commit de5ce6f3fcea97b5ed837d0254982165bd59e858 (HEAD -> master)
Author: Sandra Nataly Lima Castro <sandralima111@gmail.com>
Date:   Wed Feb 28 15:11:55 2018 -0500

    data extraction from Kaggle

commit be7e37e44f34dfec83d8fec920e7c113017bc4d3 (origin/master)
Author: Sandra Nataly Lima Castro <sandralima111@gmail.com>
Date:   Tue Feb 27 16:31:58 2018 -0500

    initial commit

$ git remote -v
origin  https://github.com/sandralima/Titanic.git (fetch)
origin  https://github.com/sandralima/Titanic.git (push)

$ git push origin master
Counting objects: 9, done.
Delta compression using up to 4 threads.
Compressing objects: 100% (9/9), done.
Writing objects: 100% (9/9), 26.57 KiB | 5.31 MiB/s, done.
Total 9 (delta 3), reused 0 (delta 0)
remote: Resolving deltas: 100% (3/3), completed with 3 local objects.
To https://github.com/sandralima/Titanic.git
   be7e37e..de5ce6f  master -> master

Name the lab-notebooks with the following convention (By Jonathan Whitmore, Data Science Team O'REILLY Safari):
[ISO 8601 date]-[DS-initials]-[2-4 word description].ipynb
example: 2015-11-21-SL-coal-predict-RF-regression.ipynb

Example of Document python code using sphinx-quickstart:

def fetch_bigtable_rows(big_table, keys, other_silly_variable=None):
"""Fetches rows from a Bigtable.

    Retrieves rows pertaining to the given keys from the Table instance
    represented by big_table.  Silly things may happen if
    other_silly_variable is not None.

    Args:
        big_table: An open Bigtable Table instance.
        keys: A sequence of strings representing the key of each table row
            to fetch.
        other_silly_variable: Another optional variable, that has a much
            longer name than the other args, and which does nothing.

    Returns:
        A dict mapping keys to the corresponding table row data
        fetched. Each row is represented as a tuple of strings. For
        example:

        {'Serak': ('Rigel VII', 'Preparer'),
         'Zim': ('Irk', 'Invader'),
         'Lrrr': ('Omicron Persei 8', 'Emperor')}

        If a key from the keys argument is missing from the dictionary,
        then that row was not found in the table.

    Raises:
        IOError: An error occurred accessing the bigtable.Table object.
    """

def public_fn_with_googley_docstring(name, state=None):
    """This function does something.

    Args:
       name (str):  The name to use.

    Kwargs:
       state (bool): Current state to be in.

    Returns:
       int.  The return code::

          0 -- Success!
          1 -- No good.
          2 -- Try again.

    Raises:
       AttributeError, KeyError

    A really great idea.  A way you might use me is

    >>> print public_fn_with_googley_docstring(name='foo', state=None)
    0

    BTW, this always returns 0.  **NEVER** use with :class:`MyPublicClass`.

    """
    return 0

FOR ubuntu:
$ git config --global user.name "Your Name"
$ git config --global user.email "youremail@domain.com"

git config --list
user.name=sandralima111
user.email=sandralima111@gmail.com

https://github.com/sandralima/MLMortgage.git

git remote -v
origin	https://github.com/sandralima/MLMortgage.git (fetch)
origin	https://github.com/sandralima/MLMortgage.git (push)

$ git status
git log --oneline
$ git add .
$ git commit -m "messsage"
$ git log
$ git push origin master

$ git init
Initialized empty Git repository in C:/Users/sandr/Documents/ENN/DLMortgage/DLMortgage/DLMortCookiecutter/mlmortgage/.git/


