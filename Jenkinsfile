// Switch to Clang to run against clang. A future iteration of the pipeline will run clang+gcc in parallel
def COMPILER_FAMILY = "GCC"

node {
  notifyPending("Starting build", COMPILER_FAMILY)

  def cmake = tool name: 'Latest', type: 'hudson.plugins.cmake.CmakeTool'
  echo "Using CMake from ${cmake}"

  def cc = tool "CC-${COMPILER_FAMILY}"
  def cxx = tool "CXX-${COMPILER_FAMILY}"
  echo "Using CC=${cc}, CXX=${cxx}"

  withEnv(["CC=${cc}", "CXX=${cxx}"]) {
    stage('Checkout') {
      try {
        notifyPending("Git checkout", COMPILER_FAMILY)
        checkout scm
      } catch (e) {
        currentBuild.result = 'FAILED'
        notifyFailure('Git checkout failed', COMPILER_FAMILY)
        throw e
      }
    }
    stage('Configure') {
      if (!fileExists('build')) {
        sh 'mkdir build'
      }
      dir('build') {
        try {
          notifyPending("Configuring with CMake", COMPILER_FAMILY)
          sh "${cmake} ../"
        } catch (e) {
          currentBuild.resuilt = 'FAILED'
          notifyFailure('CMake failed', COMPILER_FAMILY)
          throw e
        }
      }
    }

    stage('Build') {
      dir('build') {
        try {
          notifyPending("Building", COMPILER_FAMILY)
          sh 'make'
        } catch (e) {
          currentBuild.result = 'FAILED'
          notifyFailure("Build failed", COMPILER_FAMILY)
          throw e
        }
      }
    }
    notifySuccess(COMPILER_FAMILY);
  }
}

def notifyPending(description, key) {
  githubNotify context: "jenkins/${key}", account: 'otherlab', credentialsId: 'd411dfdb-4ceb-4d55-845e-46d1d40e40dc', repo: 'geode', status: 'PENDING', description: "${description}"
}

def notifyFailure(description, key) {
  githubNotify context: "jenkins/${key}", account: 'otherlab', credentialsId: 'd411dfdb-4ceb-4d55-845e-46d1d40e40dc', repo: 'geode', status: 'FAILURE', description: "${description}"
}

def notifySuccess(key) {
  githubNotify context: "jenkins/${key}", account: 'otherlab', credentialsId: 'd411dfdb-4ceb-4d55-845e-46d1d40e40dc', repo: 'geode', status: 'SUCCESS', description: 'Success!'
}
