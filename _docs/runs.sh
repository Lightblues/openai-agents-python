# merge upstream changes (upstream/main -> eason)
git remote add upstream https://github.com/openai/openai-agents-python
git fetch upstream
git merge upstream/main
# push changes to eason branch
git push origin eason
