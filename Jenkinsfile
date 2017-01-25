node {
  withVirtualenv(pwd() + "/virtualenv") {
    stage('Setup') {
      sh "python -m pip install nose numpy pytest scipy"
      tasks = [:]
      for(compiler in ['GCC', 'Clang']) {
        tasks[compiler] = {
          withNotifications("jenkins/${compiler}") {
            buildWithCompilers(compiler)
          }
        }
      }
    }
    parallel tasks
  }
}

def buildWithCompilers(family) {
  def cc = tool "CC-${family}"
  def cxx = tool "CXX-${family}"
  echo "Using CC=${cc}, CXX=${cxx}"

  def cmake = tool name: 'Latest', type: 'hudson.plugins.cmake.CmakeTool'
  echo "Using CMake from ${cmake}"

  def srcRoot = pwd()
  ws(family) {
    withEnv(["CC=${cc}", "CXX=${cxx}"]) {
      stage("Checkout ${family}") {
        checkout scm
      }
      cleanDir('build') {
        stage("Configure ${family}") {
            sh "${cmake} ${srcRoot}"
        }

        stage("Build ${family}") {
          sh 'make'
        }

        stage("Test ${family}") {
          try {
            sh "python -m nose --with-xunit"
          } finally {
            junit 'nosetests.xml'
          }
        }
      }
    }
  }
}

def cleanDir(path, cl) {
  if (fileExists(path)) {
    dir(path) {
      deleteDir()
    }
    sh "mkdir ${path}"
  }
  dir(path) {
    cl()
  }
}

def withNotifications(context, cl) {
  githubNotify context: "${context}", account: 'otherlab', credentialsId: 'd411dfdb-4ceb-4d55-845e-46d1d40e40dc', repo: 'geode', status: 'PENDING'
  try {
    cl()
  } catch (e) {
    githubNotify context: "${context}", account: 'otherlab', credentialsId: 'd411dfdb-4ceb-4d55-845e-46d1d40e40dc', repo: 'geode', status: 'FAILURE'
    throw e
  }
  githubNotify context: "${context}", account: 'otherlab', credentialsId: 'd411dfdb-4ceb-4d55-845e-46d1d40e40dc', repo: 'geode', status: 'SUCCESS'
}

def withVirtualenv(path, cl) {
  sh "virtualenv ${path}"
  withEnv(["PATH+VIRTUALENV=${path}/bin"]) {
    cl()
  }
}
