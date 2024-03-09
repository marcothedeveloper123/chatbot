# git commands

`git init` = initialize project to use git
`git add .` = add all changes to be saved
`git add *filename*` = add single file to be saved
`git ls-tree -r scrape --name-only` = list all apps git tracks
`git commit -m "message"` = save changes with message
`git push origin master` = push changes to github master
`git push origin new-branch` = push changes to github new-branch
`git pull origin master` = pull changes from github master
`git checkout -b new-branch` = create a new branch
`git checkout main` = switch to main
`git status` = check status of changes
`git log` = see all previous saved changes
`git checkout *commit hash*` = travel back to old commit
`git branch -d *branchname*` = delete branch
`git branch --delete *branchname*` = delete branch called 'new branch'
`git reset --hard; git clean -fd` = revert tracked files, remove untracked files
`git checkout main; git merge new-branch` = merge new-branch into main
`git reset <file>` = remove file from `git add`

## set up a git repository
1. `git init` = initialize project to use git

## rename the current branch
1. `git branch -m new-branch-name` = rename the current branch
2. `git push origin --delete old-branch-name` = delete the remote branch with the old name
3. `git push origin -u new-branch-name` = push the local branch with the new name to the origin repository

## revert to the latest version of a remote Git repository and discard all local changes
1. `git fetch origin` = Fetch the Latest Changes from the Remote Repository
2. `git reset --hard origin/<branch name>` = Reset Your Current Branch to Match the Remote Repository
3. `git clean -fd` = Clean Your Working Directory

## sync full repo after merging pull request in GitHub
1. `git checkout main` = Switch to your main branch
2. `git fetch origin` = Fetch the latest changes from the remote repository
3. `git merge origin/main` = Merge the fetched changes into your local branch