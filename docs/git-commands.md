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
