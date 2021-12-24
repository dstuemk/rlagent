BUILD=build
#EXPERIMENTS=experiments
#DATE=${shell date --iso-8601}
#TIME=${shell date +%H%M%S}
#EXPERIMENT_PARENT_DIR=${EXPERIMENTS}/${USER}
#EXPERIMENT_DIR=${EXPERIMENT_PARENT_DIR}/${DATE}/${TIME}

all:
	@echo "Hello Group-3"


${BUILD}:
	mkdir ${BUILD}

#doc:
#	doxygen Doxyfile

compile: ${BUILD} #doc
	cd ${BUILD} && cmake -DCMAKE_BUILD_TYPE=Release ..
	$(MAKE) -C ${BUILD} all

#experiment: compile
#	mkdir -p ${EXPERIMENT_DIR}/data
#	if [ ! -d "venv" ]; then python3 -m venv venv && venv/bin/pip install -r requirements.txt; fi
#	git ls-files | tar Tzcf - ${EXPERIMENT_DIR}/code.tgz
#	cp -r ${BUILD} ${EXPERIMENT_DIR}/${BUILD}
#	./${EXPERIMENT_DIR}/${BUILD}/group3 -cmd exec -exec train -exec_dir ${EXPERIMENT_DIR}/data
#	if ! venv/bin/python -c "import matplotlib"; then venv/bin/pip install -r requirements.txt; fi
#	venv/bin/python visualization.py ${EXPERIMENT_DIR}/data ${DATE}_${TIME} ${EXPERIMENT_PARENT_DIR} > ${EXPERIMENT_PARENT_DIR}/${DATE}_${TIME}.txt

#test: compile
#	./build/unit_test

#coverage: test
#	bash coverage.sh

#lint:
#	bash lint.sh

clean:
	rm -rf ${BUILD}
#	rm -rf doc
