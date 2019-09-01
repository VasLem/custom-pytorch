from setuptools import Command

import shlex
import subprocess
import os

WHEELHOUSE = "wheelhouse"


class _Command(Command):

    def execute(self, command, capture_output=True):
        """
        The execute command will loop and keep on reading the stdout and check for the return code
        and displays the output in real time.
        """

        print("Running shell command: %s" % command)

        if capture_output:
            return subprocess.check_output(shlex.split(command))

        process = subprocess.Popen(
            shlex.split(command), stdout=subprocess.PIPE)

        while True:
            output = process.stdout.readline()

            if output == "" and process.poll() is not None:
                break
            if output:
                print(output.strip())

        return_code = process.poll()

        if return_code != 0:
            print("Error running command %s - exit code: %s" %
                  (command, return_code))
            raise IOError("Shell Commmand Failed")

        return return_code

    def run_commands(self, commands):
        for command in commands:
            self.execute(command)


class Package(_Command):
    """Package Code and Dependencies into wheelhouse"""
    description = "Run wheels for dependencies and submodules dependencies"
    user_options = [('against-kaggle=', None, 'Save wheels, given kaggle ones.')]

    def initialize_options(self):
        """Set default values for options."""
        self.against_kaggle = True
        pass

    def finalize_options(self):
        """Post-process options."""
        assert self.against_kaggle\
            in (False, True), 'Expected boolean value for against_kaggle option!'
        pass

    def localize_requirements(self):
        """
        After the package is unpacked at the target destination, the requirements can be installed
        locally from the wheelhouse folder using the option --no-index on pip install which
        ignores package index (only looking at --find-links URLs instead).
        --find-links <url | path> looks for archive from url or path.
        Since the original requirements.txt might have links to a non pip repo such as github
        (https) it will parse the links for the archive from a url and not from the wheelhouse.
        This functions creates a new requirements.txt with the only name and version for each of
        the packages, thus eliminating the need to fetch / parse links from http sources and install
        all archives from the wheelhouse.
        """
        dependencies = open("requirements.txt").read().split("\n")
        local_dependencies = []

        for dependency in dependencies:
            if dependency:
                if "egg=" in dependency:
                    pkg_name = dependency.split("egg=")[-1]
                    local_dependencies.append(pkg_name)
                elif "git+" in dependency:
                    pkg_name = dependency.split("/")[-1].split(".")[0]
                    local_dependencies.append(pkg_name)
                else:
                    local_dependencies.append(dependency)

        print("local packages in wheel: %s" % local_dependencies)
        self.execute("mv requirements.txt requirements.orig")

        with open("requirements.txt", "w") as requirements_file:
            # filter is used to remove empty list members (None).
            requirements_file.write(
                "\n".join(filter(None, local_dependencies)))

    def restore_requirements_txt(self):
        if os.path.exists("requirements.orig"):
            print("Restoring original requirements.txt file")
            commands = [
                "rm requirements.txt",
                "mv requirements.orig requirements.txt"
            ]
            self.run_commands(commands)

    def remove_kaggle_wheels(self):
        with open(os.path.abspath('../requirements.txt'), 'r') as inp:
            kaggle_reqs = inp.readlines()
        kaggle_deps = {}
        for dependency in kaggle_reqs:
            dependency = dependency.rstrip()
            pkg_version = None
            if dependency:
                if "egg=" in dependency:
                    pkg_name = dependency.split("egg=")[-1]
                    pkg_version = None
                elif "git+" in dependency:
                    pkg_name = dependency.split("/")[-1].split(".")[0]
                    pkg_version = 'last'
                else:
                    if '==' in dependency:
                        pkg_version = dependency.split('==')[1]
                    pkg_name = dependency.split('==')[0]
            kaggle_deps[pkg_name.replace('_', '-').lower()] = pkg_version
        for wheel in os.listdir(WHEELHOUSE):
            wheel_name = wheel.split('-')[0].replace('_', '-').lower()
            wheel_version = wheel.split('-')[1]
            if wheel_name in kaggle_deps:
                if kaggle_deps[wheel_name] == wheel_version or\
                        kaggle_deps[wheel_name] == 'last':
                    os.remove(os.path.join(WHEELHOUSE, wheel))

    def run(self):
        commands = []
        commands.extend([
            "rm -rf {dir}".format(dir=WHEELHOUSE),
            "mkdir -p {dir}".format(dir=WHEELHOUSE),
            "pip wheel --wheel-dir={dir} -r requirements.txt".format(
                dir=WHEELHOUSE)
        ])

        print("Packing requirements.txt into wheelhouse")
        self.run_commands(commands)
        print("Generating local requirements.txt")
        self.localize_requirements()
        if self.against_kaggle:
            self.remove_kaggle_wheels()
        print("Packing code and wheelhouse into dist")
        self.run_command("sdist")
        self.restore_requirements_txt()


class UpdateRequirements(_Command):
    """Update requirements.txt file"""
    description = "Update requirements file using pipreqs package"
    user_options = [('against_kaggle=', True, 'Update requirements, given kaggle ones.')]



    def run(self):
        commands = ['pipreqs . --use-local --savepath requirements.txt']
        print("Trying to get requirements from package using pipreqs..")
        self.run_commands(commands)
