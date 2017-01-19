// Switch to Clang to run against clang. A future iteration of the pipeline will run clang+gcc in parallel
def COMPILER_FAMILY = "GCC"

node {
  withNotifications("jenkins/${COMPILER_FAMILY}") {
    def cmake = tool name: 'Latest', type: 'hudson.plugins.cmake.CmakeTool'
    echo "Using CMake from ${cmake}"

    def cc = tool "CC-${COMPILER_FAMILY}"
    def cxx = tool "CXX-${COMPILER_FAMILY}"
    echo "Using CC=${cc}, CXX=${cxx}"

    withVirtualenv(pwd() + "/virtualenv") {
      sh "python -m pip install nose numpy pytest scipy"
      withEnv(["CC=${cc}", "CXX=${cxx}"]) {
        stage('Checkout') {
          checkout scm
        }
        cleanDir('build') {
          stage('Configure') {
              sh "${cmake} ../"
          }

          stage('Build') {
            sh 'make'
          }

          stage('Test') {
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
