const core = require('@actions/core')
const exec = require('@actions/exec')
const artifact = require('@actions/artifact')

let errors = []

const notebook = core.getInput('notebook')

// try running the notebook
try {
  await exec.exec('papermill', ['--execution-timeout=600', '--parameters_file=./.buildkite/notebook-parameters.yml', '--log-output', notebook, notebook])
} catch (error) {
  errors.push(error.message)
}

core.startGroup("Uploading notebook as artifact")
// upload the executed notebook as an artifact, for reference
const artifactName = path.basename(notebook)
const artifactDir = path.dirname(notebook)
try {
  const artifactClient = artifact.create();
  const response = await artifactClient.uploadArtifact(artifactName, [notebook], artifactDir)
  core.debug(response)
  response.failedItems.forEach(x => errors.push(`Failed to upload "${x}"`));
} catch (error) {
  errors.push(error.message)
}
core.endGroup()

// check for errors
if (errors.length > 0) {
  const errorString = errors.join(", ")
  core.setFailed(errorString)
}
