#!/bin/bash

trap cleanup HUP PIPE TERM

set -e

function string_contains () {
  [ -z "$1"  ] || { [ -z "${2##*$1*}"  ] && [ -n "$2"  ]; };
}

remotes=`git remote -v`
echo $remotes

if string_contains "ai" "aiaia";
then
  echo "found"
else
  echo "not found"
fi

if string_contains "albert" "$remotes";
then
  echo "found github; skip configure";
else
  echo "not found github, add remotes";
  git remote add github git@github.com:albertsh10/gemini.git;
fi;

if string_contains "heng.shi" "$remotes";
then
  echo "found github; skip configure"
else
  echo "not found github, add remotes"
  git remote add origin git@git.enflame.cn:heng.shi/gemini.git
fi

echo "try to push to gitlab ~~"
git push origin master
echo "success to push to gitlab ~~"

echo "try to push to github ~~"
git push github master
echo "success to push to github ~~"
