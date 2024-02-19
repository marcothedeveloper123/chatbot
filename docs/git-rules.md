** Workflow - Solowork **
** never work directly in the main branch **
** set .gitignore to exclude .env and workspace **

** Workflow - Solowork **
`git init` = initialize project to use git
`git pull origin` = pull all changes (if working with someone else on your branch)

`git add .` = add all changes to be saved
`git commit -m "message"` = save changes with message (locally)

`git push origin master` = push changes to github master

** Workflow - Integration of changes to main (merge-reguest, pull request) **
Suggesting a pull request:
- github -> goto "Pull Request" -> "New Pull Request"

Checking and accepting/declining a pull request: 
- github: goto "Pull Request" -> "Select Pull Request"
- VSCode: Pull relevant branch on `git pull origin branch` = pull all changes (if working with someone else on your branch)
- VSCode: Test relevant branch (review code lines and test code)
- github: goto "Pull Request" -> "Select Pull Request" -> Merge and/or comment 
