findWaypoint():

	# Check if there is somthing to be avoided
	if knownObstaclesList.isEmpty() == True and collisionList.isEmpty() == True:
		return goal

	# Initiate the list of forbiddenSpaces (regions to be avoided). The waypoint must guide the robot towards a possible region among all the forbidden ones)
	forbiddenSpacesList = knownObstaclesList # even if knownObstaclesList is empty, of course

	# Add forbiddenSpaces that help the robot to avoid collisions with other robots
	if collisionList.isEmpty() == False:
		for otherRobot in collisionList:
			if areRobotsGoingToCollide(otherRobot, mySelf) == False:
				continue
			else:
				forbiddenSapcesList.append(newForbiddenSpace(collsionRadius, collisionCenter))

	# Check if there is somthing to be avoided
	if forbiddenSapcesList.isEmpty() == True:
		return goal

	# In order to avoid trying to pass among forbiddenSpaces that are to close together we have to identify cluster of forbidden spaces

	# Initiate list of clusters of forbiddenSpaces
	clustersOfForbiddenSpacesList = emptyList()

	# Build the list of clusters
	for forbiddenSpace in forbiddenSpacesList:

		if forbiddenSpace is in any cluster in clustersOfForbiddenSpacesList:
			continue

		# do a tree search in forbiddenSpacesList looking for other forbiddenSpaces forming a cluster with this forbiddenSpace. If none, this forbiddenSpace is a cluster by it self

		clustersOfForbiddenSpacesList.append(findCluster(forbiddenSpace))

	# Some clusters can be ignored, let's remove them. For the remaining clusters, let's find the information about avoiding each of their forbidden spaces
	for cluster in clustersOfForbiddenSpacesList:

		isAnyForbiddenSapceInMyWay = False

		for forbiddenSpace in cluster:
			if forbiddenSpace.isInRobotsWay(mySelf) == True:
				isAnyForbiddenSapceInMyWay = True
				break

		if isAnyForbiddenSapceInMyWay == True:

			for forbiddenSpace in cluster:

				# get angular variation and distance for avoiding this forbiddenSpace by passing by its left and its right 
				forbiddenSpace.avoidanceInfo = getAngularVariationForAvoidanceAndDist(forbiddenSpace, mySelf)

		else:
			# this cluster can be ignored in the finding of a waypoint
			clustersOfForbiddenSpacesList.remove(cluster)

	if clustersOfForbiddenSpacesList.isEmpty() == True:
		return goal

	for cluster in clustersOfForbiddenSpacesList:
		for forbiddenSpace in cluster:
			if forbiddenSpace.avoidanceInfo.distance < 0.0: #inside the obstacle
				return previousWayPoint







clustersOfForbiddenSpacesList.append(newCluster(forbiddenSpace))