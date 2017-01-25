node {
  withVirtualenv(pwd() + "/virtualenv") {
    stage('Setup') {
      sh "python -m pip install nose numpy pytest scipy"
      tasks = [:]
      for(compiler in ['GCC', 'Clang']) {
        tasks[compiler] = builderForFamily(compiler)
      }
    }
    parallel tasks
  }
}

def Closure builderForFamily(family) {
  return {
    withNotifications("jenkins/${family}") {
      buildWithCompilers(family)
    }
  }
}

def buildWithCompilers(family) {
  def cc = tool "CC-${family}"
  def cxx = tool "CXX-${family}"
  echo "Using CC=${cc}, CXX=${cxx} for ${family}"

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
  slackSend channel: 'softcad_skynet', color: '#4444ff', message: "Build Started - ${env.JOB_NAME}/${context} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)"
  try {
    cl()
  } catch (e) {
    githubNotify context: "${context}", account: 'otherlab', credentialsId: 'd411dfdb-4ceb-4d55-845e-46d1d40e40dc', repo: 'geode', status: 'FAILURE'
    slackSend channel: 'softcad_skynet', color: 'danger', message: "Build Failed - ${env.JOB_NAME}/${context} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>, <${env.BUILD_URL}/testReport|View test results>)"
    throw e
  }
  githubNotify context: "${context}", account: 'otherlab', credentialsId: 'd411dfdb-4ceb-4d55-845e-46d1d40e40dc', repo: 'geode', status: 'SUCCESS'
  slackSend channel: 'softcad_skynet', color: 'good', message: "Build Succeeded - ${env.JOB_NAME}/${context} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>, <${env.BUILD_URL}/testReport|View test results>)"
}

def withVirtualenv(path, cl) {
  sh "virtualenv ${path}"
  withEnv(["PATH+VIRTUALENV=${path}/bin"]) {
    cl()
  }
}
